# Optima contributor and agent guide

This file is the operational entry point for automated contributors. It is not
a second product manual. The canonical engineering documentation lives under
`docs/` and is built from this repository.

## Read first

Choose the smallest relevant path:

1. `docs/get-started/concepts.md` — system vocabulary and trust boundaries.
2. `docs/architecture/product-model.md` — normative proposal, crown,
   integration, and release contract.
3. `docs/architecture/slot-contract.md` — normative contribution boundary.
4. `docs/reference/state-of-record.md` — dated implementation and evidence
   status.
5. `docs/miner-guide/overview.md` — contribution workflow.
6. `docs/validator-guide/overview.md` — intake, qualification, settlement, and
   publication.
7. `docs/engine/overview.md` — chain-independent serving releases.
8. `docs/security/threat-model.md` — implemented controls and residual risk.

If a task continues earlier Codex or Claude work, follow the cross-harness
continuity instructions supplied by the environment. Historical logs route an
investigation; current code, tests, Git state, and external state remain
authoritative.

`WORKLOG.md` and `docs/WORKLOG.md`, when present, are private local working
records. They are ignored and must not be committed, linked from public docs,
or treated as production authority.

## Product invariants

- A miner proposal is hostile input, not production source.
- The validator owns the model, workload, timing, outputs, references, target
  policy, and verdict.
- A contribution changes one registered singleton/atomic target or enters the
  fenced discovery lane.
- Candidate build and execution remain outside the trusted controller in
  validator-owned, no-egress OCI lifetimes.
- CUDA graphs are part of the scored contract.
- A first PASS is `reproduction_pending`. Settlement requires an independently
  bound PASS pair and uses the lower accepted speedup.
- The resident hot-swap screen is routing-only. Its measurements cannot crown,
  settle, or authorize rewards.
- Production version-3 qualification uses two resident TP lanes, adaptive
  B/C/B′ then optional C′/B″ speed evidence, a separate eager/untimed audit
  role when registered, pristine T, and a physical-lane role swap across
  reproduction.
- Evaluation-stack settlement, incentive activation, weight publication,
  integration review, release signing, and serving are distinct authorities.
- Legacy V1 weights and inactive V2 finite debt are fenced state machines.
  Do not infer V2 activation or registered discovery promotion from implemented
  arithmetic.

If a change weakens one of these statements, it requires an explicit design and
security review—not a local implementation shortcut.

## Repository map

```text
optima/                    runtime and control-plane package
  chain/                   finalized intake, durable state, activation, weights
  eval/                    screening, qualification, OCI, evidence, scoring
  integrations/            version-pinned SGLang adapters
optima_kernels/            validator-owned reference kernel library
examples/                  miner bundles and adversarial controls
tests/                     executable contracts and regressions
docs/                      canonical documentation site
scripts/                   repository validation and reproducible studies
```

Use `docs/reference/codebase-map.md` for authority-oriented entry points.

## Development workflow

Start from a clean understanding of the worktree:

```bash
git status --short --branch
git diff --stat
```

Do not overwrite, stash, or discard unrelated user changes. Keep changes
scoped. Runtime changes should be accompanied by focused tests before the full
suite.

CPU setup and baseline validation:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cpu,dev]"
python -m pytest -q tests
```

Contributor bundle checks:

```bash
python -m optima.cli scan examples/miner_silu_torch
python -m optima.cli verify examples/miner_silu_torch \
  --device cpu \
  --dtype float32
```

Use `python -m optima.cli` for GPU work; SGLang uses spawned processes and the
module entry point preserves the required guard.

## SGLang and GPU rules

- `PINNED_SGLANG` in `optima/compat.py` is consensus-critical. An exact version
  mismatch or failed chokepoint is an error.
- The spawn-safe seam is installed in every interpreter through
  `import optima.bootstrap` in a `.pth` file.
- `optima/seams.py` is the only adapter registry. Bootstrap, activation, binding
  vocabulary, and compatibility checks derive from it.
- Adding a slot starts in `optima/slots.py`. A new SGLang chokepoint adds one
  adapter implementation and one `SeamAdapter` row; do not create a parallel
  registry.
- Block and collective contributions must satisfy graph capture/replay and
  declare the required graph metadata.
- Collective verification binds each process to its CUDA device before process
  group initialization.
- Do not mix measurements across runtime, model, image, topology, workload, or
  policy identities.

See `docs/dev/gpu-setup.md` and `docs/dev/sglang-tracking.md`.

## Trust and evidence rules

- Never import candidate Python or native extensions into the trusted
  controller.
- Treat static scanning as defense in depth, not containment.
- Do not convert infrastructure, baseline, reference, teardown, or incomplete
  evidence failures into candidate `FAIL`.
- Do not rerun only a favorable arm, splice authorities, change a threshold
  after observing a result, or replace missing evidence with logs.
- Settlement reopening and full causal regrade are different claims.
- A crown is measurement/attribution evidence. It does not approve provenance,
  maintainability, licensing, integration, or serving.
- Loading a native artifact during evaluation does not prove serving-wheel or
  release-provider closure.

See `docs/security/evidence.md` and `docs/security/isolation.md`.

## Documentation contract

Documentation ships with the code. Every pull request must make a
documentation-impact decision. Update docs in the same PR when changing:

- commands or flags;
- manifests, slots, targets, ABIs, stacks, arenas, receipts, or durable schemas;
- miner or validator workflows;
- settlement, incentives, or weight semantics;
- trust boundaries or failure behavior;
- dependencies, installation, compatibility, or releases.

Pure internal refactors, tests, and formatting may state why no docs update is
required.

Validate documentation with:

```bash
python -m pip install -r docs/requirements.txt
python scripts/check_docs.py
mkdocs build --strict
```

The checker enforces navigation coverage, internal links/anchors, repository
source links, CLI inventory, private-path exclusion, and retired-repository
removal. `site/` is generated output and is not committed.

See `CONTRIBUTING.md` and `docs/contributing/documentation.md`.

## Persistence

Committed code, tests, this file, and `docs/` are the portable context. Keep
dated empirical claims in `docs/reference/state-of-record.md` or `docs/results/`;
keep evergreen pages neutral and present-tense. Detailed chronology belongs in
Git history or `docs/history/`, not in operator and architecture pages.
