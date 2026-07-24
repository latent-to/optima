# Diagnostics by lifecycle stage

Diagnose a proposal from its last authoritative state. A local PASS cannot
override a later intake, screen, qualification, or settlement result because
each stage has different identity and evidence.

## Decision vocabulary

Optima uses three qualification grades:

| Decision | Meaning | Miner response |
|---|---|---|
| `PASS` | the bound evidence passed this stage | continue; one qualification PASS is still only `reproduction_pending` |
| `FAIL` | the candidate or its declared applicability violated a requirement | change the proposal and submit a new content identity |
| `NO_DECISION` | authority, infrastructure, conditioning, drift, or evidence was insufficient for a safe verdict | preserve the proposal identity and wait/retry under operator policy |

`NO_DECISION` is not a weak pass, and `FAIL` is not converted to a zero-scored
candidate. Non-crown screens use equivalent `pass`, `fail`, and `no_decision`
grades to derive promote, reject, retry, or hold.

## Durable intake states

The production SQLite state machine currently exposes these statuses:

| Status | What it means |
|---|---|
| `reserved` | finalized reveal admitted and waiting for a fetch lease |
| `fetching` | HTTPS fetch/extract/hash/publication work is active |
| `transport_retry` | transient transport failure is eligible for another fetch attempt |
| `published` | immutable worker publication and selected-delta identity exist; waiting for primary screens |
| `screening` | the registered arena service is running the ordered non-crown screen prefix |
| `promoted` | every non-crown screen passed; waiting to enter qualification |
| `qualifying` | one authority-bound v3 attempt—resident B/C/B′, conditional C′/B″, registered eager audit A, then pristine T—is active |
| `reproduction_pending` | one complete PASS is retained; a fresh independent screen/qualification pass is still required |
| `qualified` | two consistent independent PASS attempts are retained; settlement is separate |
| `no_decision` | retryable qualification evidence/failure product was retained |
| `held` | automatic progress stopped under retry, capacity, or safety policy; operator action is required |
| `failed` | terminal invalid/rejected/failed candidate |
| `expired` | terminal finalized-block SLA expiry; wall-clock age is not the authority |

These transitions are implemented in
[intake.py](https://github.com/latent-to/cacheon/blob/main/optima/chain/intake.py).
The command `chain-status` shows revealed chain commitments, not this private
state machine. Obtain lifecycle receipts through the operator's published
surface.

## 1. Manifest and target resolution

Start with:

```bash
python -m optima.cli scan my_bundle
```

`scan` parses paths and the component manifest and checks the source tree. It
does not reproduce production target resolution or trusted rebuild-feature
observation. For a target error, use the intake receipt and compare the parsed
manifest with the active catalog; `verify` additionally preflights variant
domains but is still not the intake resolver.

Common failures and fixes:

| Symptom | Likely cause | Fix |
|---|---|---|
| unsupported ABI | `abi_version` is not `optima-op-abi-v0` | update the bundle to the active component ABI |
| unsafe/missing path | absolute path, traversal, symlink, or undeclared file | make every declaration bundle-relative and source-only |
| competition mode mismatch | `slot`/`atomic` assertion disagrees with catalog | select the exact registered target and mode |
| target members differ | op rows do not implement the complete registered delta | use the exact singleton member or all atomic members |
| feature not allowed | `setup`, dependency patch, rebuild, override, CUDA source, or unknown extra is outside policy | remove it or use a target/lane that explicitly permits it |
| incomplete feature evidence | intake could not independently observe the rebuild feature set | use only registered rebuild declarations and complete source inventory |
| duplicate slot requires variants | repeated rows omit explicit unique `variant` | name every variant |
| overlapping domains | two variants can route the same live call | make capability domains provably disjoint |

A missing `[competition]` may still resolve for a narrow legacy singleton case,
but do not rely on that for new submissions.

### Do not collapse every local error into “scan failed”

The point at which output stops identifies the layer:

| Last output | What has actually happened | Next check |
|---|---|---|
| TOML/path exception before the bundle summary | manifest parsing or declared-path validation stopped | syntax, required fields, identifier spelling, file existence, relative path containment |
| bundle summary plus `[VIOLATIONS]` | the manifest loaded, but one declared or recursive-tree source policy failed | every printed file/line and any undeclared executable material |
| clean `scan`, then `invalid or ambiguous variant domain` in `verify` | static source is clean, but metadata/manifest eligibility cannot register deterministically | JSON types, canonical values, manifest/metadata intersection, variant overlap |
| `[SKIP] ... not a known slot` | this checkout has no such slot contract | active validator version and slot name; a skip is not evidence |
| every row N/A and `no bundle variant is applicable` | domains registered, but no row matched the selected invariant context | dtype, architecture, model, phase, topology, and required descriptor fields |
| per-shape `FAIL` | candidate ran for an applicable shape and failed its ABI/comparator | shape-specific math, output ownership, mutation, stride, metric detail |
| `NUMERICAL_PASS` with `graph=NOT_VERIFIED` | eager math passed but required capture/replay proof did not | graph phase and failure class, not numerical tolerance |

Run `scan` separately even though `verify` repeats recursive policy checks. The separate
command gives the cheapest no-import result; `verify` then tests domain registration and
candidate execution. Neither command performs production target feature resolution, so a
local clean result cannot overrule a later catalog rejection.

## 2. Capability routing

`verify` prints N/A for shapes outside a declared domain. N/A is neither failure
nor evidence that the variant works.

If every shape is N/A, check:

- canonical architecture spelling (`sm103`, not an informal GPU name);
- dtype intersection between manifest and metadata;
- exact model and runtime identifiers;
- `phase`, `quant`, `graph_mode`, TP/EP/world size;
- numeric ranges and the distinction between `num_tokens`, `q_len`, and
  `exp_tokens`;
- whether the live arena binding actually supplies every constrained field.

Unknown or missing descriptor fields fail closed. A context-applicable variant
with incomplete shape-domain coverage fails the authoritative graph veto.

## 3. ABI and numerical verification

Run the cheapest relevant check first:

```bash
python -m optima.cli verify my_bundle --device cpu --dtype float32
python -m optima.cli verify my_bundle --device cuda --dtype bfloat16
```

For collectives, reproduce the real group:

```bash
python -m optima.cli verify my_bundle \
  --device cuda --dtype bfloat16 \
  --world-size <TP> --tp-size <TP>
```

Typical failures:

- **wrong positional signature:** compare with [Kernel ABI](kernel-abi.md);
- **input mutated:** clone scratch data instead of modifying validator inputs;
- **poison remains / partial output:** write every logical output element and
  honor non-contiguous stride;
- **large elementwise error:** confirm formula, scale, mask, dtype, and model
  activation;
- **matched ratio below policy:** inspect reduction order, quantization layout,
  routing weights, and uninitialized tails;
- **top-k overlap below policy:** debug selected block sets, causality, ragged
  tail blocks, and negative-infinity masking—not just average score error;
- **one rank hangs/fails:** ensure all ranks issue collectives in the same order
  on the supplied group and do not hide an exception before a peer collective;
- **prepared representation wrong:** keep `prepare` deterministic, input-pure,
  and consistent with the live dtype/quantization binding.

A CPU pass is never a CUDA, distributed, graph, or throughput pass.

### Read a shape result as a sentence

A formatted verifier row combines five facts: applicability, validator-produced shape,
dtype/context, comparator result, and graph replay count. Start with status and detail;
do not rank candidates by `max_abs` in isolation.

For all-close contracts, one outlier is enough to fail even if the printed ratio is near
one. For matched-ratio contracts, the target-owned minimum ratio decides the result. For
MSA, `overlap` is agreement between trusted and candidate-induced top-k block sets, so
large raw score differences can be irrelevant while one ranking mistake can matter. For
low-bit cosine profiles, direction and any configured norm guard matter more than a
single maximum element error.

If only one shape fails, first compare that shape's semantic dimensions with your tiling,
mask, ragged-tail, and stride assumptions. Do not immediately narrow metadata around it.
A narrower domain is honest only when it represents a real supported algorithmic region
and still covers material calls; using eligibility to hide an implementation bug will
either leave a coverage hole or produce no marginal effect.

## 4. Graph evidence

Graph failures are classified separately:

- `graph_eager_failed`: the callable failed before capture;
- `graph_capture_failed`: capture was not legal;
- `graph_replay_failed`: capture completed but replay or replay output failed;
- `graph_applicability_failed`: observed applicability differs from the bound
  requirement;
- `graph_domain_coverage_failed`: the declared domain was not completely tested;
- missing member/variant/shape evidence or replay-count mismatch: `NO_DECISION`.

Common causes are host synchronization, data-dependent Python branching,
capture-time compilation/allocation, stale pointers, partial replay writes, or
collective ordering changes. `graph_safe: true` is only a declaration.

Do not “fix” graph failure by disabling CUDA graphs in a local profile; that changes the
serving regime. See [Graph evidence](graph-safety.md).

## 5. Chain payload, fetch, and publication

If the commitment is not accepted locally, use `chain-submit --dry-run` and
check the exact HTTPS URL and 64-character lowercase hash.

After reveal, transport failures divide into two classes:

- transient DNS/timeout/selected server failures may enter `transport_retry`;
- canonical URL, public-route, TLS, size, archive-shape, or content-hash failures
  are terminal candidate failures.

Check that:

- the hosted object is still the archive produced by `chain-package`;
- the bundle directory was not edited between package and submit;
- redirects also resolve to valid public HTTPS destinations;
- the URL has no credentials or fragment;
- the server returns the full object within current limits;
- the archive is gzip-compressed tar with one valid wrapper/root and only
  permitted identity-bearing files;
- no regular file exceeds 16 MiB, no inspectable source/configuration file
  exceeds 8 MiB, and all inspectable files remain within the 32 MiB aggregate
  budget.

Publication/storage faults after a valid fetch are validator-side
`NO_DECISION`, not candidate failure. The transport boundary is implemented in
[fetch.py](https://github.com/latent-to/cacheon/blob/main/optima/chain/fetch.py)
and immutable publication in
[publication.py](https://github.com/latent-to/cacheon/blob/main/optima/chain/publication.py).

## 6. Non-crown screens

The arena service always runs this ordered prefix:

1. `static`
2. `build`
3. `abi`
4. `graph`
5. `abbreviated_serving`

A FAIL stops at the failing stage and rejects the proposal. A retryable
`NO_DECISION` returns it to the appropriate primary or reproduction queue;
exhausted or non-retryable uncertainty is held. Passing all five only promotes
the candidate to full qualification—it does not score or crown it.

Use the last stage receipt rather than rerunning an unrelated local command.
For example, a production build failure may involve the immutable materialized
tree and pinned build image that a local combined rebuild did not reproduce.

## 7. Full qualification

Qualification aggregates mandatory graph, marginal speed, registered eager
audit A when required by the plan, and pristine T quality evidence. The v3
order is resident B/C/B′, conditional C′/B″, registered eager audit A, then
pristine T. Any FAIL makes the attempt fail; any `NO_DECISION` prevents PASS.

Speed problems:

- B and B′ disagree beyond calibrated conditioning: infrastructure/noise
  uncertainty, usually `NO_DECISION`;
- resident crossover baseline reads or physical-lane identities fail their
  bound consistency checks: authority/infrastructure uncertainty, never a
  candidate pass;
- C is not faster than the calibrated marginal bar: candidate FAIL;
- arm identities/resources differ: authority mismatch, never a valid speedup;
- candidate falls back for material calls: no positive marginal effect.

Quality problems:

- required fidelity/task metric regresses: candidate FAIL;
- stock/reference drift overlaps the calibrated boundary: `NO_DECISION`;
- referenced pristine-T identity or raw quality artifact cannot reopen:
  authority/infrastructure failure;
- a contributor-controlled quality check differs: local flags and prompts are not the
  bound reference policy.

Contributor-controlled matched A/B profiling can reproduce a mechanism, but it cannot
contest a retained validator grade or create qualification authority.

## 8. Reproduction

After the first complete PASS, status is `reproduction_pending`. The proposal
must pass a fresh independent screen and qualification attempt with matching
identity. The two retained results are consistency-checked, and the lower
speedup is used for settlement.

Do not announce a crown from `reproduction_pending`. If the second attempt is
retryable, it remains in the reproduction lane under policy. If the evidence is
inconsistent or fails, the first PASS cannot crown by itself.

## 9. Settlement, reward, and release

`qualified` means the reproduction gate passed; settlement can still wait for
older overlapping arrivals, a cohort lease, and retained-evidence reopening.
Transactional settlement may:

- crown a registered-target candidate;
- neutralize a non-winning/overlapped candidate;
- hold a candidate when authority cannot safely advance;
- award a discovery bounty instead of a standing crown.

Weight publication is a separate reconciled action. Under V1 a crown receives
decaying relative standing credit. Under the selected but inactive V2 policy,
settlement issues bounded principal that can be debited only after exact
confirmed publication. Neither generation promises a fixed per-slot or token
payout. See [Incentives](incentives.md).

Finally, crown, integration, and release are distinct. Absence from an Engine
release is not a settlement error: selected-payload preservation, surrounding packaging, review, attribution,
signing, and release construction happen later under release authority.

When reporting a problem, include the content hash, target ID, arena/evaluation
stack digest, last durable status, decision/reason, and evidence/receipt digest.
Do not include wallet secrets, private URLs, or validator filesystem paths.
