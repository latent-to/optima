# CLI reference

The CLI is a thin operator and contributor surface over Optima's typed APIs. A
command's output has only the authority listed here; human-readable output is not a
qualification, settlement, or release receipt.

```bash
python -m optima.cli <command> [options]
python -m optima.cli <command> --help
```

Use module invocation on GPU hosts because SGLang starts child processes. The
installed `optima` console script resolves to the same parser.

## Command inventory

| Command | Audience | Authority | Purpose |
|---|---|---|---|
| `slots` | all | read-only | Print the registered slot ABI |
| `compat` | operator | diagnostic | Check the installed SGLang seam surface against the pin |
| `chain-compat` | operator | diagnostic | Check the installed Bittensor SDK surface without chain access |
| `scan` | contributor | local gate | Parse a bundle and apply recursive static policy |
| `verify` | contributor | local gate | Check declared slot behavior against validator-owned references |
| `chain-package` | contributor | packaging | Build a canonical archive and print its content hash |
| `chain-submit` | contributor | chain mutation | Commit a bundle hash and HTTPS fetch location through timelock reveal |
| `chain-status` | all | read-only | Inspect public subnet, registration, and reveal state |
| `chain-register` | operator | chain mutation | Burn-register a hotkey and run the SDK preflight |
| `chain-validate` | validator | production intake | Consume finalized reveals; a deployment may inject qualification services |
| `chain-archive-schema3-hold` | validator | durable state transition | Terminally archive one exact legacy schema-3 reproduction hold |
| `chain-incentive-shadow` | policy operator | signer-free evidence | Project explicit synthetic registered-CROWN debt against finalized membership |
| `chain-incentive-composition-shadow` | policy operator | signer-free evidence | Project explicit synthetic CROWN and discovery debt against finalized membership |
| `chain-activate-incentives` | policy operator | wallet-free durable transition | Atomically activate one independently approved campaign/composition |
| `set-weights` | signer | legacy production control plane | Reconcile the journaled V1 projection, including bounded burn bootstrap/watch operation |
| `mint-push-credentials` | operator | weight-share push auth | Create/rotate HMAC secrets for eval → serve-weights |
| `push-weight-offer` | eval | peer weight distribution | Build V1/V2 offer and HTTP-push; never chain-publishes |
| `serve-weights` | weights gateway | peer weight distribution | Serve/store the offer; optional authenticated PUT from eval |
| `follow-weights` | signer | peer weight publication | Fetch the shared offer and publish through the commit-reveal reconciler |
| `set-debt-weights` | validator | active-V2 production control plane | Publish, confirm, and debit the next gapless finite-debt boundary |
| `model-provision` | release operator | production artifact | Seal model bytes into a content-addressed publication and receipt |
| `release-verify` | release consumer | production verification | Reopen a signed Engine release under an externally trusted key |
| `release-context` | release consumer | production build input | Materialize a deterministic OCI context from a verified release |

There is no local command that grants qualification or settlement authority. Complete
engine qualification begins at the deployment-injected arena boundary; a local evaluator
or JSON ledger is not an alternate production interface.

## Contribution commands

### `slots`

```bash
python -m optima.cli slots
```

Reads the registered `SLOTS` table without importing contribution code or requiring a
GPU. See the [slot catalog](slots-table.md).

### `scan`

```bash
python -m optima.cli scan path/to/bundle
```

Loads `manifest.toml` as data and recursively applies the Python policy to declared and
vendored `.py` files. Manifest-declared CUDA sources and dependency patches are admitted
as separate reviewed-build tiers; undeclared executable files, binaries, and symlinks are
rejected. Static scanning is defense in depth; a clean result does not make contribution
code trusted.

### `verify`

```bash
# CPU contract smoke
python -m optima.cli verify examples/miner_silu_torch \
  --device cpu --dtype float32

# CUDA contract and graph verification
python -m optima.cli verify path/to/bundle \
  --device cuda --dtype bfloat16 --model MiniMax-M3

# Distributed verification at the arena topology
python -m optima.cli verify path/to/bundle \
  --device cuda --world-size 4
```

Options are `--dtype`, `--device`, `--seed`, `--world-size`, `--tp-size`, and
`--model`. The verifier rejects ambiguous variant domains, applies the model-specific
slot profile, and spawns workers for candidate execution. Collective slots use the
requested rank count; a host without enough CUDA devices falls back to CPU/Gloo unless
`--device cuda` makes the requirement explicit.

Verification proves only the exercised component contract. It does not establish model
integration, serving throughput, pristine quality, isolation, independent reproduction,
or settlement.

## Submission commands

### Package

```bash
python -m optima.cli chain-package path/to/bundle --out bundle.tar.gz
```

The command writes one canonical wrapper archive and prints the deterministic content
hash of the extracted bundle tree. Host the exact archive at a stable HTTPS URL. The URL
is transport; the hash is proposal identity.

### Submit

```bash
python -m optima.cli chain-submit path/to/bundle \
  --url https://artifacts.example/bundle.tar.gz \
  --network <network> --netuid <netuid> \
  --wallet <wallet> --hotkey <miner-hotkey> \
  --blocks-until-reveal 10
```

`chain-submit` re-hashes the local bundle before constructing the payload. Use
`--dry-run` to print the payload without signing or submitting it. Validators act only
on finalized, valid reveals and independently fetch, extract, and re-hash the hosted
archive.

### Inspect public chain state

```bash
python -m optima.cli chain-status \
  --network <network> --netuid <netuid> \
  --wallet <wallet> --hotkey <hotkey>
```

The wallet arguments add registration information. This command does not expose the
validator's private intake, screening, qualification, or settlement database.

### Register a hotkey

```bash
python -m optima.cli chain-register \
  --network <network> --netuid <netuid> \
  --wallet <wallet> --hotkey <hotkey>
```

This is a chain mutation and may burn registration cost. It checks for existing
registration first, then runs the SDK preflight.

## Validator commands

### `chain-validate`

The stock entrypoint supports complete finalized intake without a GPU service:

```bash
python -m optima.cli chain-validate \
  --network <network> --netuid <netuid> \
  --intake-only --once
```

Intake mode persists finalized order, hardened fetch and re-hash results, private
retention, immutable worker publication, and copy disposition. Storage and loop controls
are `--intake-db`, `--private-root`, `--publication-root`, `--interval`, and `--once`.

Qualification requires a deployment-owned `ArenaServiceRegistry` plus a registered
`--arena-id`. The repository does not construct a production provider from shell text.
A reviewed deployment wrapper calls `cmd_chain_validate(args,
arena_registry=registry)` or `run_validator(...)` after creating the registry. Invoking
the stock module without `--intake-only` refuses to run because no registry was injected.

### `chain-archive-schema3-hold`

```bash
python -m optima.cli chain-archive-schema3-hold \
  --network <network> --netuid <netuid> \
  --intake-db chain_intake/intake.sqlite3 \
  --reservation-id <reservation-id> \
  --reason "reviewed migration disposition"
```

This is a terminal, evidence-preserving disposition for one exact legacy
schema-3 single-PASS hold. It does not qualify, reproduce, release, crown, or
publish weights. Current-schema work must use the normal authority path.

### `set-weights`

```bash
python -m optima.cli set-weights \
  --network <network> --netuid <netuid> \
  --half-life-blocks <blocks> \
  --discovery-lifetime-blocks <blocks> \
  --discovery-pool-ppm <ppm> \
  --refresh-blocks <blocks> \
  --dry-run
```

The command reopens settled state and computes a pure global projection. `--dry-run`
creates no publication intent and submits nothing; stable stdout reports the projection
digest, `status=dry_run`, `chain_matches`, and `submitted=False`. It does not print the
projected UID/weight vector or the complete projection inputs. A live reconciliation
journals intent before submission and confirms it only through later chain observation.
`--release-hold REASON` appends an audited release of the held publication and does not
submit. `--reconcile-only --validator-hotkey <hotkey>` confirms or releases without
constructing a signer.

`--burn-hotkey <hotkey>` is available only while retained authority is completely
uncrowned and has no active reward claim or V2 composition. It projects the complete pool
to that registered bootstrap identity and fails closed as soon as normal reward authority
exists. `--watch --interval <seconds>` runs repeated reconciliations with bounded retry
rules; it cannot be combined with dry-run, reconcile-only, or hold release. Remove the
burn hotkey before restarting after the first CROWN.

Every non-hold `set-weights` pass also writes the exact publishable projection to
`<intake-db>.current_weights.json` (or `--weight-offer-path`) and, when configured,
asynchronously to a swappable object store (`--object-store-provider hippius|s3|minio|local`).
Prefer the eval/serve/follow split below when eval must not hold a chain-signing
weight path: `push-weight-offer` → `serve-weights` → `follow-weights`.

### `mint-push-credentials` / `push-weight-offer` / `serve-weights` / `follow-weights`

```bash
python -m optima.cli mint-push-credentials --path /secret/push-credentials.json

python -m optima.cli serve-weights \
  --object-store-provider hippius \
  --object-store-bucket optima-weights \
  --push-credentials /secret/push-credentials.json \
  --network <network> --netuid <netuid> \
  --wallet default --hotkey weights-gateway \
  --host 0.0.0.0 --port 8080

python -m optima.cli push-weight-offer \
  --intake-db chain_intake/intake.sqlite3 \
  --network <network> --netuid <netuid> \
  --url http://weights-gateway:8080 \
  --push-credentials /secret/push-credentials.json \
  --attribution-hotkey <placeholder-ss58> \
  --half-life-blocks <blocks> \
  --discovery-lifetime-blocks <blocks> \
  --discovery-pool-ppm <ppm>

python -m optima.cli follow-weights \
  --url http://weights-gateway:8080 \
  --network <network> --netuid <netuid> \
  --wallet default --hotkey follower \
  --refresh-blocks <blocks> \
  --expected-authority <weights-gateway-hotkey> \
  --watch
```

`push-weight-offer` is the eval path: it builds a V2 debt/composition offer when
incentive composition is active (else legacy V1), and HTTP-PUTs it. It never
opens a weight-signing wallet or calls `set_weights`. Credentials resolve from
`--push-credentials`, else `OPTIMA_WEIGHT_PUSH_CREDENTIALS` (JSON path), else
`OPTIMA_WEIGHT_PUSH_KEY` (+ optional `OPTIMA_WEIGHT_PUSH_CREDENTIAL_ID`).
`serve-weights` exposes `GET /v1/current-weights` (permit + hotkey signature) and
optional `PUT /v1/current-weights` (same credential resolution). `follow-weights`
rebinds the offer to the follower hotkey and publishes through
`reconcile_weight_publication` / commit-reveal. Debt-lane offers carry the full
`DebtWeightPublicationBinding` so follower `weights_ppm` match the economic
projection. Provider swap is config-only via `--object-store-provider` /
`OPTIMA_OBJECT_STORE_*` (optional dep: `pip install -e ".[object-store]"`, boto3
Apache-2.0). See
[Settlement and weights](../validator-guide/settlement-and-weights.md#shared-current-weights-endpoint).

### Incentive shadows

```bash
python -m optima.cli chain-incentive-shadow \
  --network <network> --netuid <netuid> \
  --policy core-policy.json \
  --claims-fixture synthetic-core-claims.json \
  --expected-policy-digest <sha256> \
  --expected-claims-digest <sha256> \
  --output core-shadow-receipt.json

python -m optima.cli chain-incentive-composition-shadow \
  --network <network> --netuid <netuid> \
  --core-policy core-policy.json \
  --core-claims-fixture synthetic-core-claims.json \
  --discovery-policy discovery-policy.json \
  --discovery-claims-fixture synthetic-discovery-claims.json \
  --expected-core-policy-digest <sha256> \
  --expected-core-claims-digest <sha256> \
  --expected-discovery-policy-digest <sha256> \
  --expected-discovery-claims-digest <sha256> \
  --output composition-shadow-receipt.json
```

Both commands require explicitly synthetic fixtures, bind exact finalized membership,
construct no wallet, and never submit. Their receipts establish deterministic projection
against observed membership; they do not provide review, activation, settlement,
publication, or debit authority.

### `chain-activate-incentives`

```bash
python -m optima.cli chain-activate-incentives \
  --network <network> --netuid <netuid> \
  --intake-db chain_intake/intake.sqlite3 \
  --core-policy core-policy.json \
  --composition-policy composition-policy.json \
  --approval approval.json \
  --expected-approval-digest <independently-recorded-sha256>
```

Activation is wallet-free. It validates and atomically binds the exact finalized cursor,
retained arena, stack, catalog/family roster, membership, reserve, audit controls, policy,
and independent approval. The implemented path accepts exactly one immutable MiniMax-M3
campaign. It does not sign or publish weights.

### `set-debt-weights`

```bash
python -m optima.cli set-debt-weights \
  --network <network> --netuid <netuid> \
  --intake-db chain_intake/intake.sqlite3 \
  --wallet <wallet> --hotkey <validator-hotkey> \
  --refresh-blocks <blocks> \
  --dry-run
```

This command is valid only after V2 activation. It projects the next exact gapless policy
boundary, journals before signing, confirms through authoritative readback, and debits
claims only after confirmation. Delayed boundaries preserve nominal order and catch up no
faster than the policy cadence. `--reconcile-only`, `--validator-hotkey`, and
`--release-hold` provide signer-free recovery surfaces analogous to the V1 publisher.

## Environment checks

```bash
python -m optima.cli compat
python -m optima.cli chain-compat
```

`compat` checks the exact pinned SGLang version plus registered imports and signatures. A
version mismatch is a failing result. `chain-compat` checks the Bittensor SDK API used by
Optima without connecting to a network.

## Release commands

### Provision model bytes

```bash
python -m optima.cli model-provision \
  /srv/models/model /srv/optima/model-publication \
  --expected-content-digest <sha256> --workers <n>
```

The result is an immutable content-addressed model tree and receipt.

### Verify a release

```bash
python -m optima.cli release-verify /srv/optima/releases/<digest> \
  --expected-public-key <ed25519-public-key> \
  --descriptor-digest <expected-digest>
```

The expected public key is an external trust input. A key discovered only inside the
release cannot authenticate its signer.

### Materialize a container context

```bash
python -m optima.cli release-context \
  /srv/optima/releases/<digest> ./context \
  --expected-public-key <ed25519-public-key> \
  --descriptor-digest <expected-digest>
```

The command reopens the complete signed publication before writing a deterministic OCI
context. Release construction and signing remain programmatic APIs; there is no public
`release-create` command.

## Exit behavior

Commands use a non-zero status for parser errors, explicit refusals, and failed local
checks, but exit `0` is only command completion. It is not always business-state success:
for example, `set-weights` can complete with a durable `pending` publication that still
requires later chain observation. Automation must inspect the typed status and retain the
receipts or durable records emitted by the authoritative subsystem. Console prose and
process status alone are never settlement or publication evidence.

Source: [`optima/cli.py`](https://github.com/latent-to/cacheon/blob/main/optima/cli.py).
