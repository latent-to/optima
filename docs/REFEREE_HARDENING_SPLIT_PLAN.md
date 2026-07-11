# Referee hardening extraction plan

The frozen `codex/referee-hardening-donor-20260711` worktree is a valuable exploratory donor, not a
merge candidate. It was frozen from pre-extraction main `203bb559`; it must never be merged or
cherry-picked wholesale. Clean extraction PRs start from current main or the preceding accepted
slice. A clean branch means clean review history and bounded scope—not reimplementation.

The GitHub PR numbers and the architectural PR numbers in this document are intentionally
different. GitHub PRs #31–#37 are merged sub-slices of architectural PR 1, not architectural
PRs 1–7. GitHub PR #38 committed this plan/Gate 0, and PR #39 closes architectural PR 1 with
the distributed verifier plus live-collective parity. At #39's exact code head `e7b7ddcb`,
the clean extraction relative to `203bb559` is production +6,343/-476, tests +6,490/-125,
and docs +1,521/-34 including the final receipt update.

## Gate 0 — freeze the product contract

`docs/PRODUCT_CONTRACT.md` is the architectural authority. In particular:

- miner proposals, economic crowns, integrated contributions, and engine releases differ;
- complete isolated engines are the execution boundary;
- the smallest validator-controlled adoptable delta is the reward boundary;
- the untrusted evaluation stack and reviewed release stack differ;
- a pristine untimed reference, not an untrusted incumbent, owns quality grading;
- slots are a fast path, not the limit of permissible optimization;
- bounded engine/source deltas can discover new targets without a referee PR.

Changing one of these is a product decision, never an incidental implementation detail.

## Preserve the donor before extraction

1. Stop all writers and GPU runs.
2. Create a private raw checkpoint containing the binary tracked diff, nonignored
   untracked paths, selected ignored ledgers, and a SHA-256 inventory.
3. Secret-scan it. Exclude `.pass`, wallet material, caches, bytecode, model data, and
   credentials.
4. Freeze it on `codex/referee-hardening-donor-20260711` with one clearly labeled
   `archive: ... (not for merge)` commit.
5. Add an annotated donor tag and an offline Git bundle. Push only after the secret scan;
   never open the donor as a PR.
6. Verify the donor commit in a detached worktree and record its actual status. The current
   teacher-v2 checkpoint is compile-clean, not test-certified.
7. Make the donor read-only. Perform extraction in new worktrees based on `origin/main` or
   the preceding accepted PR.

Every donor file or hunk receives exactly one disposition in the donor map:

- `PORT`: transplant substantially intact;
- `ADAPT`: retain the mechanism but change its product boundary;
- `DELETE`: superseded or duplicative;
- `DEFER`: useful experiment outside the current architecture;
- `VENDOR`: isolated provenance/licensing commit.

Complete new files may be restored from the donor. Mixed existing files are applied
hunk-by-hunk. The donor commit is never cherry-picked wholesale.

## Provenance prerequisite

Before PR 2, land a provenance-only commit for the pinned Moby seccomp profile, required
MiniMax/SGLang overlay assets, `NOTICE`, licenses, hashes, and package-data declarations.
Vendor material never pads an ordinary implementation PR.

## Eight focused PRs

The ceilings below are review budgets, not quotas. Exceeding one requires splitting the PR
or recording a written exemption before implementation continues.

| PR | Scope | Production/test ceiling |
|---|---|---:|
| 1 | Typed contribution contracts and target catalog | 2.5k / 1.5k |
| 2 | Streaming isolated engine executor | 4.5k / 2.2k |
| 3 | Evaluation stack, release identity, and marginal assembly | 2.5k / 1.2k |
| 4 | Replayable qualification and pristine reference authority | 3.5k / 1.8k |
| 5 | Fenced bounded-engine discovery lane | 1.5k / 0.6k |
| 6 | Finalized chain intake and immutable worker publication | 2.0k / 0.9k |
| 7 | Transactional settlement, stack transition, and global weights | 2.0k / 1.0k |
| 8 | Reproducible releases, joined proof, and deletion closure | 1.0k / 0.5k |

Guidance total: about 19.5k authored production and 9.7k test additions, excluding the
separate vendor commit. The line budget is subordinate to correctness but prevents silent
return to another 52k-line monolith.

The budget is cumulative as well as per-PR. PR #39 contains two explicit review commits:
the distributed verifier (+931/-87 production, +660/-2 tests) and live parity
(+1,354/-266 production, +1,402/-143 tests). The whole PR against its `main` parent is
production +2,243/-311, tests +2,062/-145, and docs +178/-15 including this final receipt update.
Production stays under the 2.5k PR-1 guidance; retained adversarial tests exceed the 1.5k
guidance by 562 additions. This is the written exemption: four real failure classes were found
only by that matrix (input/reference mutation, fixed-value replay, cross-shape capture state,
and live producer/consumer identity), duplicate cases were consolidated, and independent size
review found no remaining safe production deletion.

Cumulative clean extraction at the same code head is production +6,343/-476 and tests
+6,490/-125: 32.5% and 66.9% of the whole-plan budgets, 43.9% combined. Architectural PR 1
therefore closes at 2.54x/4.33x its original production/test ceiling. This is materially larger
than planned, not hidden: 96.9% of #39 production additions are in the five owning verifier/live
ABI modules, and no PR 2/3, chain, settlement, economics, OCI, or SM120 kernel-porting scope
entered. Every subsequent merge still records its classified delta, deletes the legacy path it
actually supersedes, or explains why none exists. Coverage and static analysis nominate cleanup;
they are not deletion authority for GPU-only or intentionally public paths.

Implementation order is not a literal 1→2→3 pipeline. After PR 1 completes, land the pure
product core of PR 3 first (`PR 3a` below), then PR 2, then the small runtime bridge (`PR 3b`).
This makes the product-defining stack shape the executor input without pretending that stack
schemas alone can produce an isolated or authoritative crown.

### PR 1 — Typed contribution contracts and target catalog

Port/adapt typed tensor outputs, capability domains, variants, validator fallback,
capture/replay, completed-versus-fallback receipts, setup gating, scanning, and exact
singleton/atomic target semantics.

`TargetCatalog` defines semantic regions, members, overlap/displacement rules, and allowed
contribution features. It does not decide crowns or emissions. The three proposal tiers are:

1. known target replacement;
2. registered cross-target/atomic delta;
3. bounded engine/source delta routed to discovery.

Exit:

- two independent variants plus off-domain stock fallback ingest without `optima/` edits;
- offline/live ABI parity and graph replay pass;
- one singleton and one atomic target resolve canonically regardless of manifest order;
- component receipts are diagnostic, never external crown authority;
- no OCI, chain, stack economics, or whole-system title enters this PR.

The final contribution-foundation work is deliberately split into two review boundaries:

1. distributed verifier correctness (trusted pre-candidate snapshots, fresh-value graph
   replay, same-process multi-shape capture, real topology/dtype, strict diagnostic wire);
2. live collective parity (one canonical descriptor/allocation projection, variant-keyed MoE
   prepare state, and identical deep export/consume selection).

Those reproduced P1s, the bounded adversarial pass, and the confirmation review are complete on
PR #39; PR 1 freezes when it merges. A generic typed-input
IR, arbitrary-topology framework, secure crown authority inside candidate workers, and SM120
ports are explicitly not PR 1 work.

### PR 2 — Streaming isolated engine executor

Port the disposable prebuild, streaming session, external host timing, charged
conditioning, device/runtime/model/source receipts, non-root/read-only/no-egress/seccomp
launch, watchdogs, immutable publication, and exact cleanup. Its input becomes a generic
content-addressed `EngineLaunchSpec` that references a validator-materialized engine tree, not a
miner bundle assumption.

Exit:

- the trusted controller imports/loads no candidate Python or native code and owns clocks;
- the executor can launch any validator-materialized engine tree;
- no-op B/C/B' mechanics and tamper/import/timeout/resource/cleanup negatives pass on RTX;
- shared validation is extracted, then the legacy one-shot HMAC/result-file worker, dead
  close protocol, and production candidate-audit paths are deleted;
- wallet keys and chain clients never enter the GPU executor.

### PR 3 — Evaluation stack, release identity, and marginal assembly

Add the small genuinely missing layer:

- `TargetSpec`/catalog digest;
- `ContributionRef` and immutable attribution;
- `EvaluationStackManifest` for hostile hill-climbing incumbents;
- `EngineReleaseManifest` for reviewed Optima source only;
- deterministic composite-bundle materialization;
- exact-one-delta B/C/B' construction;
- overlap, supersession, dependency, rollback, and last-known-good transactions.

Reuse the existing manifest, registry, shared-module loader, rebuild patchers, immutable
artifact publication, and OCI path. The materializer copies selected declared closures,
rewrites paths, and emits one validator-generated manifest/rebuild recipe; it is not a new
plugin framework or optimizer IR.

Split this layer at the isolation dependency:

- **PR 3a, before PR 2:** catalog snapshot/digest, `ContributionRef`, evaluation/release
  manifests, deterministic composite-tree materialization, exact-one-delta cohort planning,
  overlap/supersession/rollback/LKG semantics, and a content-addressed
  `MaterializedEngineTree`. It is CPU-authoritative for identity and may use RTX only for a
  development assembly/routing receipt. It cannot crown.
- **PR 3b, after PR 2:** the narrow bridge from frozen marginal arm plans and materialized
  trees to `EngineLaunchSpec`, followed by no-op and portable-target B/C/B' lifecycle proof on
  RTX. Authoritative quality still waits for PR 4's pristine reference.

Assembly must deterministically namespace generated Python and native module identities, or
fail closed on collisions. Current loaders derive identities from filename stems, so merely
copying independent contributions into separate directories is insufficient; manifest order
must never choose a winner. The namespace transform is part of the materialized-tree digest.

Exit:

- B/B' bind byte-identical incumbent stack/materialization digests;
- every `C1..Ck` differs from the frozen cohort incumbent by exactly one catalog target or
  registered atomic replacement;
- singleton MSA and atomic fused-epilogue fixtures assemble, route, regress, and roll back;
- miners never repackage incumbent contributions;
- a crown can update only the evaluation incumbent; release state changes only through an
  explicit integration record;
- duplicate permanent whole-serving reward families are impossible by construction.
- cohort construction, secret candidate ordering, incumbent rebasing, and deterministic
  winner selection are explicit rather than arrival-time side effects.

### PR 4 — Replayable qualification and pristine reference authority

Adapt the teacher-v2 and retained-evidence work to PR 3 identities. B/C/B' remain timed
isolated evaluation-stack engines. After candidate destruction, a separate untimed
pristine T arm teacher-forces sealed trajectories and runs hidden tasks. B' is never
trusted merely because it is the incumbent.

Exit:

- proposal, catalog, incumbent stack, candidate stack, exact delta, materialized trees,
  reference manifest, arena, and runtime identities are all bound;
- rollout KL and teacher NLL are distinct typed values;
- quality PASS/FAIL/NO_DECISION propagates through `EvalOutcome` and retry authority;
- selection uses retained post-commit entropy or commitment/reveal;
- raw token numerators, elapsed intervals, conditioning constituents, trajectories,
  top-k frames, teacher traces, and hidden-judge evidence reopen and recompute;
- hidden corpus/judge secrecy and rotation exist for calibrated arenas;
- RTX stock/no-op/sabotage controls pass; B300 thresholds remain explicitly unfrozen.
- one T engine can regrade a sealed finalist cohort without any candidate process surviving.
- every empirical crown gate references a content-addressed `CalibrationManifest` binding its
  derivation algorithm, reference, exact arena/runtime/hardware/workload identity, raw evidence,
  seeds, and positive/negative controls; missing, stale, mismatched, or tampered calibration is
  `NO_DECISION`;
- in-engine `audit.py` remains useful sampled slot evidence, but is never the external grading
  authority because it shares the candidate process. Pristine T owns final quality authority.

### PR 5 — Fenced bounded-engine discovery lane

Port the bounded SGLang source-overlay and activation machinery, but make it a discovery
proposal rather than a second permanent whole-engine title. This is how a serious engineer
can create a new fusion window, layout, schedule, or integration point without changing
the trusted referee first.

Exit:

- inspectable source deltas can change allowlisted model/data-plane execution and declare
  applicability, dependencies, conflicts, and build inputs;
- API/tokenizer/sampler/result/timing/service-control surfaces remain excluded;
- the validator supplies the exam and builds/evaluates the complete isolated engine;
- a win produces a bounded discovery reward plus a promotion/integration record, not an
  equal permanent fork title;
- one promotion path creates a new catalog target, atomic target, or reviewed engine change.

### PR 6 — Finalized chain intake and immutable worker publication

Port finalized reveal history, strict payloads, HTTPS/SSRF/redirect/socket/archive/hash
hardening, and copy/provenance primitives. Keep intake separate from settlement.

Exit:

- every finalized reveal durably reserves true chain/event order before transport;
- unresolved earlier priority blocks conflicting later settlement until explicit expiry;
- private `0600` intake is rehashed into a separate immutable worker tree with `0555`
  directories and `0444` files for UID/GID 65532;
- only the submitted delta is fingerprinted; canonical base is excluded;
- exact whole-delta identity or symmetric containment is authoritative, while shared
  fragments are advisory;
- testnet 307 reveal -> publication -> non-root OCI -> restart succeeds without weights.
- epoch cutoff, per-hotkey/target admission bounds, and oldest-finalized priority keep a
  continuous submission stream from making the GPU queue unbounded.

### PR 7 — Transactional settlement, stack transition, and global weights

Port the evidence-bound score state machine, retry leases, validator holds, causal pending
recovery, exclusive writer, global projection, and publication intent/pending/held/confirmed
journal. Adapt titles into contribution/crown/integration/evaluation-stack states.

Production authority uses a transactional store behind one deterministic state-machine
interface; the donor JSON ledger is migration/test input, not multi-process production
authority. A single control-plane signer owns weights; GPU executors own no wallet.

Before this PR starts, freeze a short emissions-policy contract covering relative improvement,
time decay, specialist compatibility, atomic/discovery bounties, and the prohibition on assuming
argmax-only winner-take-all rewards. Settlement implements that contract; it does not invent it.

Exit:

- invalid standing contributions HOLD without title or weight mutation;
- retirement, neutralization, crown, adoption, and stack transition are explicit events;
- chain authority is refreshed before reconciliation and after submission;
- one canonical reward family prevents packaging-based double payment;
- crash/restart, concurrency, stale state, retry, and publication reconciliation pass;
- dry-run global weights bind exact retained evidence; a real extrinsic requires a genuine
  current-schema crown and is never fabricated for testing.
- queue/epoch state and the frozen incumbent digest recover transactionally after restart.

### PR 8 — Reproducible releases, joined proof, and deletion closure

Port/adapt source/model/referee release and provisioning tools. Produce a signed,
chain-independent Optima Engine from reviewed `EngineReleaseManifest` entries only.

Exit:

- engine build and serve smoke work with Bittensor dependencies, credentials, and miner
  hosting removed;
- source/wheel/container double-build reproducibly with exact license, SBOM/provenance,
  upstream, model, overlay, seccomp, and reference identities;
- every donor hunk is PORT/ADAPT/DELETE/DEFER/VENDOR and all superseded paths/fixtures are
  gone;
- RTX runs the complete non-emitting chain -> publication -> materialized marginal B/C/B'
  -> T reference -> evidence -> crown -> evaluation-stack transition -> global dry-run ->
  restart path;
- only B300 remains for SM103/CuTe, NVLink/P2P/custom all-reduce, consensus calibration,
  real candidate performance, independent-seed crown, and crown-backed CR publication.

## Git and review discipline

- PR 1 starts from `origin/main`; each successor targets the frozen parent while stacked.
- At most one implementation PR and its immediate successor are active.
- Each PR contains a few semantic commits, not one donor-sized transplant.
- Every commit is tested from a clean detached worktree with `pyenv activate sn120`.
- Rebase reruns CPU/packaging tests. GPU receipts survive only when every bound digest is
  identical; otherwise rerun them.
- Tests earned by observed failures are retained; repeated schema fixtures and tests for
  deleted paths are consolidated or removed.

## Evaluation latency gate

The full bracket is a final referee, not the first hill-climbing step. Run the first synthetic
arrival-rate/cohort simulation immediately after PR 3a, because cohort/bookend amortization
affects identities bound by PRs 3 and 4. Refine it after the executor exists. Before PR 8 closes,
the RTX and B300 arenas must measure and freeze:

- intake/build/correctness capacity;
- resident or abbreviated screen throughput and false-negative policy;
- candidate cohort size and bookend drift as a function of elapsed time;
- TP4 dual-half concurrency and half-swap overhead where supported;
- T batching throughput for sealed finalist trajectories;
- p50/p95 reveal-to-decision latency under a registered arrival-rate stress test.

No optimization may obtain a crown from a cheap tier, but no valid submission should pay
four model loads before it has demonstrated enough signal to justify them.

## Finite audit rule

Each PR receives exactly:

1. one implementation review;
2. one bounded adversarial review against its frozen threat matrix;
3. fixes for accepted findings;
4. one confirmation pass.

A late finding reopens a PR only for a reproducible P0/P1 fail-open, a direct product
contract violation, or failure of an explicit exit criterion. Hypothetical P2/P3 work goes
to the backlog. New feature scope requires removing equivalent work or opening another PR.

## Endpoint

Gate 0 is on main and architectural PR 1 is complete on GitHub PR #39. Architectural PR 3 remains
the main genuinely missing product layer; PR 2 and later layers still have substantial donor
material but are not merged. After #39 the bounded merge sequence is: PR 3a; vendor provenance;
PR 2; PR 3b; PR 4; PR 5; PR 6; PR 7; PR 8. Split PR 2 or PR 4 only when reviewability genuinely
requires it; larger merge units retain their exit tests and finite reviews rather than cutting
corners or repeating the seven-micro-PR cadence for every layer.

The refactor's operational acceptance test is:

1. either real campaign bundle enters without a referee code edit;
2. the validator materializes the frozen incumbent plus exactly that submitted delta;
3. bounded isolated B/C/B' execute, with cohort/bookend amortization measured;
4. after candidate destruction, pristine T independently grades sealed quality evidence;
5. testnet intake and transactional settlement update the evaluation stack and global dry-run;
6. reviewed winners assemble into a reproducible, chain-independent serving release;
7. RTX proves the complete portable path, while B300 is reserved for SM103/NVLink/P2P,
   real-candidate performance, calibration, and crown evidence.

The refactor ends when all eight exit criteria and the product-contract acceptance test pass.
The endpoint is explicit invariants, clean reproducible receipts, and documented residuals—not
the impossible claim that no future reviewer can imagine another risk.
