# Discovery lane

Discovery is the bounded path for an optimization that cannot be expressed as
one current registered target. It evaluates an exact source proposal without
letting that proposal invent a permanent target, mutate the standing stack, or
become release code automatically.

Use discovery for a genuinely cross-cutting SGLang change, a candidate new
seam, a proposed atomic boundary, or a reviewed engine change. If a registered
singleton or atomic target already expresses the delta, use the component ABI
instead.

## Component competition versus discovery

| Question | Component proposal | Discovery proposal |
|---|---|---|
| What is being replaced? | one registered singleton or complete registered atomic target | an exact bounded patch that does not fit a current target |
| ABI | `optima-op-abi-v0` registered slot rows, including sealed direct artifacts | `optima-discovery-abi-v1` closed patch manifest |
| Source shape | manifest, declared entry/metadata, and allowed advanced inputs | exactly the manifest and declared unified diffs |
| Runtime authority | validator dispatches through a registered slot seam | validator applies a sealed overlay through fixed discovery process roles |
| Economic result | standing target crown may participate in emissions | one bounded, non-renewable, expiring bounty claim when enabled |
| Effect on incumbent | settlement may replace the target entry in the evaluation stack | ephemeral C arm; no standing stack entry |
| Product effect | none automatically; integration is separate | none automatically; requested promotion is only a review request |

Discovery is not a “more permissive bundle.” Its patch surface is broader than one
callable, but its inventory, touched paths, symbols, build profile, and activation are
more tightly closed. The candidate still does not choose the model, evaluator, launch
controls, network policy, or result.

### Classify the idea, not the implementation language

The same language can appear in either lane. A Triton function implementing
`activation.silu_and_mul` is a component proposal because its semantics fit that slot. A
small Python diff changing scheduler batching is discovery because scheduler behavior is
outside every slot. Conversely, a CUDA implementation is not automatically discovery:
declared `.cu` source can belong to a component target that allows its registered rebuild
feature.

The same rule applies to native publication paths. A `cutlass.cute.cubin.v1` export remains
a component proposal when it implements an existing registered target and that target
allows the provider's manifest and rebuild features. A policy-constrained FlashInfer
dependency patch can also remain a component advanced input. Its sealed overlay carries
the validator-declared `sm100` and `sm103` modules, builds each architecture in a separate
hermetic child before any FlashInfer import, and installs only the row matching the live
device. Neither mechanism grants authority to patch arbitrary engine control flow.

Use these examples:

- a faster algorithm behind the exact `attention.sdpa` signature: component singleton;
- two implementations of the same slot split at `q_len = 256`: component variants with
  disjoint domains;
- a coupled replacement of the two members already registered as
  `collective.moe_epilogue.v1`: component atomic target;
- a new fusion requiring a scheduler change and a new callable seam: discovery request
  for `new_singleton` or `atomic_target`, depending on the proposed policy outcome;
- a patch maintainers might adopt directly without creating a competitive seam:
  discovery request for `reviewed_engine_change`; and
- a useful one-off experiment with no desired persistent boundary: discovery
  `bounty_only`.

If the active catalog already represents the complete semantic delta, discovery is the
wrong lane even when a patch would be easier to write. The component boundary is what
makes overlap, fallback, attribution, and standing rewards well-defined.

## Discovery is a separate ABI

A discovery proposal is **not** an op bundle, and
`[competition] mode = "system"` is not a valid discovery schema. Its closed manifest uses
`optima-discovery-abi-v1` and contains only exact text patches plus applicability
and build-profile claims.

An illustrative schema fixture is:

```toml
bundle_id = "proposal-one"
abi_version = "optima-discovery-abi-v1"
build_profile = "minimax-m3-rtx-sm120-tp8-v1"
patches = ["patches/change.patch"]
dependencies = ["cuda13"]
conflicts = []
requested_promotion = "new_singleton"

[applicability]
arenas = ["minimax-m3-rtx-tp8-v1"]
models = ["minimax-m3-nvfp4"]
architectures = ["sm120"]
tensor_parallel_sizes = [8]
```

Those identifiers come from a test fixture and do not assert that a public
operator currently offers that profile. A real proposal must name an exact
operator-registered build profile whose arena, model, architecture, TP size,
SGLang pin, features, and build-input digests fall inside the proposal's
applicability.

Arrays must be sorted and unique. `patches` must be non-empty and contain only
canonical `.patch`/`.diff` paths. The proposal tree may contain exactly
`manifest.toml` and the declared patch files—no README, kernel directory,
binaries, or undeclared notes.

The parser and frozen-inventory checks are in
[discovery.py](https://github.com/latent-to/cacheon/blob/main/optima/discovery.py).

## Requested promotion is not authority

`requested_promotion` must be one of:

- `new_singleton`
- `atomic_target`
- `reviewed_engine_change`
- `bounty_only`

This records the miner's intended disposition. It does not register a target,
approve shipping, or force the validator to adopt the requested boundary.

For `new_singleton` and `atomic_target`, the request describes what a later policy review
might register; it is not available to this proposal as a target. For
`reviewed_engine_change`, it identifies a possible product review path, not an instruction
to merge. `bounty_only` explicitly asks for no promoted subject. In every case, a
qualification result and a promotion review are different retained records.

A qualified discovery result gets a win record. A later validator review may
create a separate promotion record with a review digest and, except for
`bounty_only`, a named subject. Creating a target or integrating source requires
an independent product/policy change.

## The source surface is closed

Discovery accepts exact unified diffs against the validator-pinned SGLang tree.
The default policy allows inspectable source only in named inference regions,
including selected layers, model executor/model files, memory-cache and overlap
code, plus an enumerated scheduler file set.

Sensitive surfaces are explicitly excluded. Path and added-source policy blocks
sampler, logits/logprobs, tokenizer/detokenizer, API/network, metrics/profiling,
results, and timing/control imports. Changes to `scheduler.py` are further
limited to enumerated semantic function/method regions; the AST outside those
regions must remain unchanged.

The exact current allowlists, symbol regions, suffixes, and forbidden names are
the `DEFAULT_DISCOVERY_POLICY` in
[discovery.py](https://github.com/latent-to/cacheon/blob/main/optima/discovery.py).
Do not rely on a prose list when preparing a patch.

Diff application is exact: no fuzz, offset search, deletion, rename, copy, or
binary patch. Added Python must parse, and the complete result is rechecked for
dynamic/aliased access to excluded surfaces.

## Build and execution model

The validator freezes the manifest and patch bytes, resolves them against a
registered build profile, and policy-checks them against the exact pinned source
tree. It then:

1. copies the complete pinned SGLang package into a private native-artifact
   stage;
2. applies the already validated patches;
3. inventories the complete resulting package and touched files;
4. publishes and reopens an immutable overlay;
5. activates that sealed overlay only through the validator's fixed process-role
   machinery inside the isolated engine.

This overlay is a frozen patched SGLang source tree. It is not the FlashInfer dependency
overlay or a direct CUBIN publication; those component mechanisms retain their own closed
build, architecture, ABI, and runtime-selection policies.

The discovery C arm is ephemeral and bound to the incumbent stack/tree,
proposal digest, policy digest, build-profile digest, and overlay identity. It
does not add a standing entry to the incumbent evaluation stack.

Candidate execution still uses the validator-owned OCI service, no-egress
policy, and resource controls. V3 loads the exact incumbent and discovery
candidate once onto disjoint resident TP lanes, serializes B/C/B′ and
policy-authorized C′/B″ reads, runs registered eager audit A, tears down
candidate lifetimes, and then obtains pristine T quality authority. Discovery
code does not get to choose its own evaluator or report its own score.

## Qualification and reward

Discovery has no crownable target. Passing the same non-crown and independent
qualification discipline can make it eligible for transactional discovery
settlement, but not for a registered-target crown.

Legacy V1 can retain one non-renewable, expiring discovery claim in its separate
pool. The selected but inactive V2 composition instead retains a qualified
discovery as review-pending and currently implements only a bounded
`bounty_only` disposition. Its durable path rejects registered promotion until
typed cross-lane work identity, target registration, and fresh-crown linkage
exist. Re-submission cannot renew a proposal already retained or awarded; exact
duplicates are terminally disposed before settlement.

If policy review later registers a singleton or atomic target, a fresh
component proposal must use that target's ABI and pass qualification,
reproduction, and settlement. The discovery proposal does not silently become
the target crown.

See [Incentives](incentives.md) for both reward generations and the
[product model](../architecture/product-model.md) for the release boundary.

## Development inspection

There is currently no dedicated public `discovery-validate` CLI command. You
can exercise the closed parser directly from a development checkout:

```bash
python - <<'PY'
from optima.discovery import inspect_discovery

proposal = inspect_discovery("my_discovery")
print(proposal.proposal_digest)
print(proposal.manifest.requested_promotion)
PY
```

This freezes and checks the proposal inventory and unified-diff structure. Full
path/symbol validation additionally needs the operator's exact policy and pinned
SGLang tree; build-profile, overlay, isolation, and qualification remain
validator authority.

Treat this local inspection as the discovery analogue of parsing plus freezing a
component bundle, not as `verify`. It can prove that the inventory and diff syntax are
closed and produce a stable proposal digest. It cannot prove the patch applies to the
operator's pinned tree, touches only allowed semantic regions, builds in the registered
profile, preserves serving behavior, or improves performance.

## Submission

Before submitting, confirm that the operator has enabled discovery intake and
published the exact build profile you request. Package and submit the proposal
through the same content-addressed HTTPS timelock path described in
[Submitting](submitting.md):

```bash
python -m optima.cli chain-package my_discovery \
  --out dist/my_discovery.tar.gz

python -m optima.cli chain-submit my_discovery \
  --url https://downloads.example.org/optima/my_discovery.tar.gz \
  --netuid <NETUID> --network <NETWORK> \
  --wallet <WALLET> --hotkey <HOTKEY> --dry-run
```

An operator without the requested profile or discovery arena cannot safely
evaluate the proposal; do not assume normal component availability implies
discovery capacity.
