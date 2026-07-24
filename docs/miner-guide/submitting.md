# Submitting a proposal

Production submission is a hotkey-signed, timelock commit-reveal containing a
content hash and an HTTPS fetch URL. The chain carries a reference, not the
archive bytes.

The miner-side implementation is in
[submit.py](https://github.com/latent-to/cacheon/blob/main/optima/chain/submit.py),
and the canonical payload is defined by
[payload.py](https://github.com/latent-to/cacheon/blob/main/optima/chain/payload.py).

## The identity chain

Submission binds several related but non-interchangeable values:

```text
bundle files --SHA-256--> content_hash --inside canonical payload--> commitment
      |                         |                              |
 exact proposal bytes      fetch authentication          finalized arrival order
```

The URL says where to fetch; it does not define the proposal. The content hash says what
bytes must be recovered; it does not prove the bytes are eligible or correct. The
finalized reveal gives arrival authority; it does not prove that a fetch or qualification
succeeded. Later selected-delta, stack, launch, evidence, and settlement digests bind the
same proposal into progressively narrower decisions.

Keep the printed content hash with your bundle and operator receipts. A bundle name such
as `alice-silu-v1` is for humans and is not enough to diagnose which bytes were evaluated.

## Before you sign

Confirm all of the following against the operator's current announcement:

- network, netuid, active arena, target catalog, and evaluation stack;
- registered target and required target mode;
- submission window, timelock policy, and any admission limits;
- the designated version of the submission terms;
- how the operator publishes intake and qualification status.

The repository's
[submission terms](../legal/submission-terms.md)
are currently marked draft and become binding only when an operator announcement
designates a version. Read the designated terms before signing. Ensure you own
or can grant the required rights to every submitted source fragment.

Register the miner hotkey first if needed:

```bash
python -m optima.cli chain-register \
  --netuid <NETUID> --network <NETWORK> \
  --wallet <WALLET> --hotkey <HOTKEY>
```

Registration needs coldkey authorization. Normal `chain-submit` is hotkey-signed
and does not need the coldkey.

## 1. Freeze and check the bundle

Use an explicit contribution target and source-only contents. Then run the
development checks appropriate to the target:

```bash
python -m optima.cli scan my_bundle
python -m optima.cli verify my_bundle --device cuda --dtype bfloat16
```

Inspect the tree for credentials, caches, generated binaries, model data,
machine paths, unsupported licenses, and stale result metadata. `scan` and
`verify` are diagnostics; they do not pre-approve intake.

## 2. Package exactly the identity-bearing files

```bash
python -m optima.cli chain-package my_bundle \
  --out dist/my_bundle.tar.gz
```

The command prints a lowercase SHA-256 content hash. That hash identifies the
canonical extracted bundle tree, not the gzip byte stream. The packager includes
exactly the regular files covered by bundle identity.

Do not edit `my_bundle` after packaging. `chain-submit` re-hashes the directory;
if it changes while the hosted archive does not, the validator will reject the
fetch as a content mismatch.

For extra confidence, extract the hosted object into a clean temporary location and run
`chain-package` against that root, then compare its printed content hash. The wrapper
directory and gzip encoding are not the identity; the sorted relative paths and file bytes
are. Never “refresh” a stable URL with revised content after committing the old hash.

## 3. Host over stable public HTTPS

Upload the archive to a URL such as:

```text
https://downloads.example.org/optima/my_bundle.tar.gz
```

Production URLs must be canonical HTTPS with a public-routable host. Credentials
in the URL, fragments, plaintext HTTP, local files, and private/loopback
destinations are rejected. Fetch retains TLS hostname verification and requires
TLS 1.2 or newer.

Keep the exact object available long enough for reveal, finalized intake, and
configured transport retries. Avoid a short-lived signed URL. The revealed URL
is public chain data, so never embed a secret in it.

The production transport accepts gzip-compressed tar only. Current bounds include
a 64 MiB archive, 256 MiB extracted content, 4,096 logical members, 16 MiB per
regular file, 8 MiB per inspectable source/configuration file, 32 MiB across all
inspectable files, bounded extension metadata, at most five redirects, and one
60-second absolute DNS/TLS/transfer/extraction deadline. The validator re-hashes
the safely extracted identity-bearing tree. See
[fetch.py](https://github.com/latent-to/cacheon/blob/main/optima/chain/fetch.py).

## 4. Dry-run the chain payload

```bash
python -m optima.cli chain-submit my_bundle \
  --url https://downloads.example.org/optima/my_bundle.tar.gz \
  --netuid <NETUID> --network <NETWORK> \
  --wallet <WALLET> --hotkey <HOTKEY> \
  --blocks-until-reveal <BLOCKS> \
  --dry-run
```

Check that the printed `content_hash` equals the package result. The payload is
canonical JSON with exactly three fields:

```json
{"v":1,"h":"<64-lowercase-hex>","u":"https://.../my_bundle.tar.gz"}
```

The production payload cap is 1,024 bytes. A refused dry run has not signed or
sent anything.

## 5. Submit the timelock commitment

Run the same command without `--dry-run`:

```bash
python -m optima.cli chain-submit my_bundle \
  --url https://downloads.example.org/optima/my_bundle.tar.gz \
  --netuid <NETUID> --network <NETWORK> \
  --wallet <WALLET> --hotkey <HOTKEY> \
  --blocks-until-reveal <BLOCKS>
```

The SDK encrypts the payload for automatic reveal after the requested timelock.
This is not the old local salt/round simulation. The finalized reveal position
provides the consensus arrival order used by intake.

You can inspect public chain state with:

```bash
python -m optima.cli chain-status \
  --netuid <NETUID> --network <NETWORK> \
  --wallet <WALLET> --hotkey <HOTKEY>
```

`chain-status` shows subnet and revealed-commitment state. It does not read the
validator's private SQLite intake lifecycle; use the operator's published
status/receipt surface for later stages.

## What happens after reveal

The authoritative path is staged:

1. A finalized valid reveal is reserved in durable SQLite intake.
2. The validator fetches the HTTPS archive into private storage, safely
   extracts it, and verifies the committed content hash.
3. It republishes an immutable worker-readable tree and fingerprints the
   selected delta.
4. Target resolution and the `static → build → ABI → graph → abbreviated
   serving` non-crown screens run through a registered arena service.
5. A promoted candidate receives a complete isolated v3 qualification attempt:
   resident B/C/B′, conditional C′/B″, registered eager audit A, then pristine
   T.
6. One PASS moves the proposal to `reproduction_pending`. It has **not** crowned.
7. A second independent matching PASS completes qualification; the lower of the
   two reproduced speedups is retained.
8. Transactional settlement may crown, neutralize, or hold the qualified
   candidate according to the frozen target/stack authority and competing
   cohort.
9. Weight projection is a separate audited control-plane action.

There is no universal completion time. Finality, queue bounds, arena capacity,
retry policy, reproduction scheduling, and settlement cadence are operator
configuration.

### Follow one proposal through the states

Suppose the revealed content hash is `H` and its target is
`activation.silu_and_mul`:

1. `reserved` means the finalized arrival has a durable intake row. The proposal has not
   been fetched, so local correctness results are not relevant to its current wait.
2. `fetching` either produces an authenticated private tree or a transport result. A
   timeout may become `transport_retry` for the same `H`; a hash mismatch is a terminal
   candidate problem because the bytes at the URL are not `H`.
3. `published` means an immutable worker tree and selected-delta identity exist. It does
   not mean candidate Python has passed any screen.
4. `screening` records the ordered stage receipts. If ABI fails, changing a local file
   cannot repair `H`; fix the source, package a new hash, and submit it as a new proposal.
   A validator storage fault should instead produce uncertainty for operator retry, not a
   fabricated candidate failure.
5. `promoted` means all five non-crown screens passed and capacity may now be
   spent on resident B/C/B′, conditional C′/B″, registered eager audit A, then
   pristine T. It carries no speed score and no reward.
6. `reproduction_pending` means the first full attempt passed. Continue to describe the
   object as a proposal awaiting independent reproduction.
7. `qualified` means two matching passes exist. Settlement still reopens evidence and
   considers priority/overlap before creating a crown.
8. A settlement crown is an economic record for this target and stack authority. It is
   still not an Engine release.

At each step, ask whether the next action changes proposal identity. Retrying fetch,
reopening retained evidence, or rerunning an independent attempt can preserve `H` under
operator policy. Editing source, metadata, the manifest, or any other identity-bearing
file necessarily creates a new hash and returns to submission.

### What to record from the operator

When the operator exposes receipts, retain at least the content hash, finalized arrival
position, target ID, arena and evaluation-stack digests, selected-delta digest, last
durable status, decision/reason, and evidence or receipt digest. These let both sides
distinguish “wrong bundle,” “same bundle under a different arena generation,” and
“infrastructure could not decide.”

`chain-status` alone cannot supply those lifecycle fields. It observes public subnet and
reveal state, while production intake and qualification state live in validator storage.
Use the operator's designated status surface rather than assuming absence from
`chain-status` output means rejection.

A crown remains separate from source integration and an Optima Engine release.
Submitting does not cause the validator to publish miner code as a release.
Reward generation and confirmed publication are described in
[Incentives](incentives.md).

## Production authority boundary

The supported submission path is `chain-package` followed by `chain-submit`. Finalized
chain intake, SQLite qualification state, transactional settlement, and journaled weight
publication are separate validator authorities. No local ledger or contributor-side
score can create those records.

If the proposal stalls or fails, map its reported lifecycle state through
[Diagnostics](diagnostics.md).
