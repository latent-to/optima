# A validator's first hour

This checklist proves that the repository, local contract checks, and public intake
surface work on your host. It does **not** create a production arena, run authoritative
qualification, or make the host safe for arbitrary candidate code.

There are consequently two useful meanings of “first hour”:

1. **Public host acceptance**, which you can complete from this repository and ends with
   a restart-safe intake-only pass.
2. **Production commissioning**, which continues only after your deployment supplies a
   reviewed `ArenaServiceProvider`, worker fleet, evidence store, calibration, signer
   policy, and release procedures. The second path is a checklist, not a hidden turnkey
   command.

Record the exact source revision, Python build, installed dependency set, host identity,
and command output as commissioning evidence. A later green run on different bytes is not
evidence for this host.

## 1. Create a clean development environment

Optima requires Python 3.10 or newer. The core package deliberately does not pin PyTorch,
because GPU installations must match the host's CUDA and SGLang environment.

For a CPU-only contract check:

```bash
git clone https://github.com/latent-to/cacheon.git
cd cacheon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[cpu,dev,release]"
```

For a GPU host, install the reviewed CUDA-compatible PyTorch and SGLang stack first, then
install Optima editable without asking pip to replace those packages. Follow
[GPU setup](../dev/gpu-setup.md).

## 2. Inspect the contracts

```bash
python -m optima.cli slots
pytest -q
```

`slots` prints the live ABI catalog. The test suite is the executable behavior
contract; a passing CPU suite does not prove a GPU topology or OCI deployment.
The CPU extras do not install SGLang, so a clean CPU-only environment should not
run `compat` and expect green output.

Capture the environment alongside the results:

```bash
git rev-parse HEAD
python --version
python -m pip freeze
```

On a host where the reviewed SGLang stack is installed, run:

```bash
python -m optima.cli compat
```

`compat` checks installed SGLang integration points and exits nonzero if the
package is absent, its version differs from the exact reviewed pin, or a chokepoint is
broken. It is only the static seam canary.
Before treating the installed runtime as an arena-qualified pin, follow the
empirical gates and dated status linked from
[SGLang compatibility](../dev/sglang-tracking.md).

If the Bittensor SDK is installed, also run:

```bash
python -m optima.cli chain-compat
```

This introspects the SDK methods Optima uses. It does not connect to a network.

## 3. Verify a known bundle locally

```bash
python -m optima.cli scan examples/miner_silu_torch
python -m optima.cli verify examples/miner_silu_torch \
  --device cpu --dtype float32
```

CPU verification checks manifest routing and op-level correctness against the validator
reference. It does not predict GPU speed, CUDA-graph behavior, end-to-end quality, or a
crown.

Run diagnostics only on code you are prepared to execute. `verify` loads candidate code
in spawned workers rather than the trusted CLI process, but the public diagnostic path is
not the production OCI authority.

## 4. Prepare state directories

Choose separate durable locations for private intake, immutable worker publication, and
the SQLite database. A local development layout is:

```text
chain_intake/
├── intake.sqlite3
├── private/
└── worker/
```

Create the database parent and private root with an owner-private umask. The publication
root itself is managed by the immutable publisher:

```bash
umask 077
install -d -m 0700 chain_intake chain_intake/private
```

Do not pre-populate these paths with shared or symlinked content. Production code checks
owner, mode, link shape, and content identity. Plan SQLite-aware backups and disk/inode
monitoring before daemon mode.

The parent of `intake.sqlite3` must be exactly validator-owned mode `0700`; an existing
database and its lock/WAL sidecars must be regular, owner-owned files with safe modes. If
this validation fails, correct the filesystem deployment. Do not weaken the check.

## 5. Exercise finalized intake

After configuring a real chain endpoint and netuid, run a single intake-only pass:

```bash
optima chain-validate \
  --netuid <NETUID> \
  --network <WSS_ENDPOINT> \
  --intake-db chain_intake/intake.sqlite3 \
  --private-root chain_intake/private \
  --publication-root chain_intake/worker \
  --intake-only \
  --once
```

This command reads finalized history, reserves new arrivals, performs HTTPS fetch and
hash verification, and publishes safe immutable copies. It does not need a wallet and
cannot qualify or settle while `--intake-only` is set.

Interpret the one-pass summary by stage:

| Counter | Meaning |
|---|---|
| `seen` | Newly read finalized reveal records, including invalid payload dispositions |
| `reserved` | New rows durably placed in canonical priority order |
| `published` | New hash-verified immutable worker publications |
| `copies` | Later authoritative submitted-delta copies demoted against an earlier miner |
| `rejected` | Terminal attributable intake failures |
| `held` | Work requiring operator or bounded retry disposition |
| `screens`, `decisions`, `settlements` | Disabled in intake-only mode |

Run the identical command a second time. With no newly finalized reveals, it should not
refetch or republish the same arrival. That checks the finalized cursor and idempotent
publication join; it does not yet exercise interruption during a live fetch.

## 6. Optional GPU diagnostics

On a configured GPU development host, use [Running diagnostics](running-evals.md) to
compare a bundle against a local model. Keep CUDA graphs enabled when testing graph-bound
targets, and treat all results as diagnostic.

Do not make a diagnostic pass crownable by wrapping it in a shell script. Production
qualification requires the registered arena, immutable engine trees, OCI execution,
pristine reference, retained evidence, and independent reproduction described in
[Qualification](qualification.md).

## 7. Commission the arena integration

This step begins deployment-owned work. Before removing `--intake-only`, freeze and
review all of the following as one service identity:

1. exact runtime, base engine, validator overlay, worker distribution, model revision,
   model content, GPU architecture, topology, GPU count, and TP size;
2. prompt-corpus digest, seed scheme, decode and long-prefill regimes, and exact shapes;
3. queue, screen, qualification, cohort, age, and retry bounds;
4. five ordered non-crown screen timeouts plus resident-screen swap, canary, waiver,
   and lifetime policy;
5. provider implementation digest and qualification-policy digest; and
6. adaptive resident speed policy, two non-overlapping physical TP lanes, audit-only
   plan, calibration, pristine reference, evidence-root, entropy, hidden judge, OCI
   executor, and absolute-deadline authorities used by `build_qualification`.

Deployment code then constructs the provider, `ArenaService`, and closed
`ArenaServiceRegistry`, and calls `run_validator(...)` with the exact registered arena
ID. Start with `once=True` under supervision. The following is deliberately only the
composition boundary—the `provider` object is not supplied by this repository:

```python
from optima.arena_service import ArenaService, ArenaServiceRegistry
from optima.chain.validator_loop import run_validator

service = ArenaService(reviewed_manifest, provider)
registry = ArenaServiceRegistry((service,))

result = run_validator(
    subtensor,
    netuid,
    intake_db="chain_intake/intake.sqlite3",
    private_root="chain_intake/private",
    publication_root="chain_intake/worker",
    arena_registry=registry,
    arena_id=reviewed_manifest.runtime.arena_id,
    intake_only=False,
    once=True,
)
```

Do not substitute a shell command, dynamic import path, or fake provider that declares
success. A commissioned provider must return real typed screen evidence and construct the
resident B/C/B′/(C′/B″), audit, and pristine-T qualification work.

## 8. Observe one complete reservation lifecycle

Before daemonizing, retain evidence for one deliberately controlled admission and verify
these transitions in order:

```text
reserved → fetching → published → screening → promoted → qualifying
         → reproduction_pending → screening → promoted → qualifying
         → qualified → leased settlement → crowned/held/discovery_bounty
```

A controlled negative should terminate as `failed`; an induced validator-side outage
should become retryable or `NO_DECISION`/`held`, never an economic loss. Confirm that the
second PASS uses the reproduction lane and different authority, selection, attempt,
report, and evidence identities. Confirm that both evidence roots reopen before a
settlement candidate is admitted.

Structural qualification fixtures can validate transitions, independence checks, and
evidence reopening. They cannot satisfy the production-provider, GPU-performance, or
calibration requirements of this commissioning step.

For resident authority, also confirm that timed GPU work never overlaps between lanes,
both lanes prove quiescence before audit/T, borderline evidence alone adds C′/B″, and the
reproduction exact-swaps the physical baseline and candidate lane roles.

## 9. Commission the signer separately

For legacy V1, only after the store contains a genuine current-schema crown and complete
active arena state should the signer run `optima set-weights --dry-run`. The explicit
`--burn-hotkey` bootstrap is valid only in an all-uncrowned database and disables itself
when any real economic authority exists. Use the exact emissions-policy arguments
approved by the validator set.

The signer and validator loop open the same single-writer SQLite authority. Coordinate a
clean reconciliation window between validator passes. A dry run creates no publication
journal intent. A real run must persist intent before signing and is not confirmed until
the exact recipient set, normalized values within the fixed verifier tolerance,
and a sufficiently new `last_update` are read back from chain.

V2 commissioning is a separate migration. Run both signer-free shadows, independently
approve the exact policy/arena/campaign/membership/audit bundle, execute the wallet-free
`chain-activate-incentives` cutover, and only then commission `set-debt-weights`. Claims
must remain unchanged until an exact finalized readback is confirmed. Follow
[Emissions policy](../reference/emissions-policy.md); the repository currently retains
no live V2 activation or publication receipt.

## 10. Commission release operations independently

If this validator also operates the release plane, prove that an active crown can be
reopened, reviewed source can be promoted into an integrated contribution, a model tree
can be sealed, and an existing signed release can be verified against an externally
pinned public key. Release construction, signing, registry publication, and serving-host
launch are reviewed programmatic deployment APIs, not public CLI commands.

Do not call the validator commissioned merely because unit and slice tests pass. The
release plane requires the complete
[publication checklist](../engine/release-workflow.md#publication-checklist): clean-wheel
closure, provider-specific native proof, builder-authenticated reproducible OCI outputs,
closed effective serving policy, serving at the approved topology, and bound serve
receipts. The evaluation plane separately requires a reviewed production
`ArenaServiceProvider`; neither boundary is supplied by passing the library test suite.

## Production-readiness checklist

Before enabling full validation, an operator still needs to supply and review:

- a digest-pinned OCI runtime and seccomp policy;
- trusted native prebuild and runtime hosts with durable lease recovery;
- sealed model bytes and exact runtime/topology identities;
- a representative workload and frozen calibration;
- an `ArenaServiceProvider` and closed `ArenaServiceRegistry`;
- an evidence-retention root and restore procedure;
- queue, timeout, retry, and disk-capacity monitoring;
- a separate hotkey-only weight signer and publication journal procedure; and
- release-key, registry, integration-review, and serving-fleet processes if operating
  the release plane.

Also require explicit incident procedures for transport exhaustion, screen/qualification
retry exhaustion, interrupted controller state, evidence-root loss, settlement lease
expiry, stale projections, held publication journals, signer mismatch, release-key
rotation, and serve-receipt failure. Each procedure must preserve the original record and
append a reviewed disposition; deleting SQLite rows is not recovery.

The repository enforces the interfaces between these parts; it does not provision the
fleet for you.

## Source anchors

- [Package metadata](https://github.com/latent-to/cacheon/blob/main/pyproject.toml)
- [CLI parser](https://github.com/latent-to/cacheon/blob/main/optima/cli.py)
- [Compatibility canary](https://github.com/latent-to/cacheon/blob/main/optima/compat.py)
- [Chain SDK canary](https://github.com/latent-to/cacheon/blob/main/optima/chain_canary.py)
