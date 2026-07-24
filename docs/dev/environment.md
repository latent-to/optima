# Development environment

Optima supports a small CPU-only contributor loop and a separate GPU/runtime
loop. Production qualification adds a third boundary: reviewed deployment code
plus isolated OCI workers. Do not flatten those environments into one trust
level.

## Choose the environment by question

| Question | Minimum environment | What a green result means |
|---|---|---|
| Does the manifest parse and resolve to one target? | CPU contributor environment | Structure and registered capability policy are coherent |
| Does an entry satisfy its tensor contract? | CPU for CPU-capable examples; otherwise GPU | The component matches the trusted reference for the exercised cases |
| Is a CUDA implementation graph-safe and dispatchable? | Exact GPU, Torch, CUDA, and pinned SGLang environment | The registered component and seam gates passed on that topology |
| Does a proposal improve serving without quality loss? | Validator-owned arena and isolated complete engines | One qualification attempt produced reopenable evidence |
| Can reviewed source ship? | Release environment plus model/native publications and signing authority | A signed release can be reopened under external trust inputs |

These rows are cumulative only in tooling, not in authority. A GPU component
verification is stronger than a CPU unit test for that component, but it is
still not a qualification. Likewise, a crown is measurement authority rather
than release approval.

## CPU-only setup

Use Python 3.10 or newer:

```bash
git clone https://github.com/latent-to/cacheon.git
cd cacheon

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[cpu,dev,release]"
```

Run the deterministic local loop:

```bash
python -m optima.cli slots
python -m optima.cli scan examples/miner_silu_torch
python -m optima.cli verify examples/miner_silu_torch \
  --device cpu --dtype float32
pytest -q
```

CPU verification is useful for manifest, scanner, and numerical contract work.
It does not establish CUDA graph behavior, collective correctness, serving
performance, or qualification.

### Read the first loop correctly

`slots` answers what the validator currently exposes. `scan` answers whether
the bundle is structurally admissible without importing it. `verify` imports
and executes the selected entry against validator-owned inputs and references.
The commands intentionally form a ladder: do not debug numerical output until
the manifest and target resolve cleanly.

The faithful SiLU example should verify successfully. The companion
`examples/miner_silu_broken_torch` implementation deliberately omits the SiLU
operation and should fail correctness:

```bash
python -m optima.cli verify examples/miner_silu_broken_torch \
  --device cpu --dtype float32
```

A failing negative control is evidence that the local verifier can distinguish
the semantic operation from a cheaper look-alike. It is not an installation
failure to “fix.”

## Release dependencies

Release signing and verification use the `release` extra, included in the setup
above. If you began with the smaller `cpu,dev` environment, add it with:

```bash
python -m pip install -e ".[cpu,dev,release]"
```

Keep signing keys out of the repository, virtual environment, shell history,
test fixtures, and generated OCI contexts. Tests should use ephemeral keys.

## Working conventions

- Invoke GPU commands as `python -m optima.cli ...` so SGLang child processes
  resolve the same installed package.
- Treat `optima/slots.py`, `optima/target_catalog.py`, stack manifests, and
  qualification schemas as contracts. Change tests and documentation with
  them.
- Use temporary directories for intake, model, native, and release tests.
  Durable production SQLite files are not development scratch space.
- Never run an untrusted bundle on a workstation or control-plane host.
  `scan` is not a sandbox.
- Preserve unrelated local changes; generated caches, build trees, receipts,
  and credentials should remain untracked.

## A practical validation ladder

Use the narrowest test that can falsify the change, then widen the boundary:

1. **Pure contract tests** for parsing, canonical identity, schemas, target
   resolution, and economics.
2. **Component execution** for reference comparisons, variants, output
   ownership, and negative controls.
3. **Integration tests** for stack planning, Engine-tree materialization,
   intake restart, evidence reopening, and release construction.
4. **GPU seam tests** for the exact SGLang pin, architecture, graph replay,
   collectives, model, and tensor-parallel topology.
5. **Arena qualification** for load-once, disjoint-lane, serialized resident
   B/C/B′ timing with conditional C′/B″, registered eager audit A, and
   candidate-free pristine T quality. Only this layer can create qualification
   authority.

If a lower layer fails, stop there. A model benchmark cannot excuse an invalid
manifest, and a fast candidate cannot excuse a failed trusted reference.

## Focused tests

The test suite is organized by authority boundary. Examples:

```bash
pytest -q tests/test_static.py tests/test_target_catalog.py
pytest -q tests/test_stack_manifest.py tests/test_engine_tree.py
pytest -q tests/test_qualification.py tests/test_qualification_runner.py
pytest -q tests/test_chain_intake.py tests/test_weight_publication.py
pytest -q tests/test_release.py tests/test_release_host.py
```

Use the complete suite before merging cross-boundary changes. GPU-dependent
tests may skip on a CPU machine; a local pass does not replace the relevant
hardware campaign.

### Match tests to the authority you changed

| Change | Start with | Widen to |
|---|---|---|
| Manifest or bundle hashing | static/manifest and target-catalog tests | example bundles and full suite |
| Slot ABI or verification | slot, tensor, and verifier tests | graph replay and the supported GPU matrix |
| Target overlap/composition | target catalog and stack planning | settlement and emissions projection |
| Intake or retry state | chain intake and validator-loop tests | restart, evidence, and weight reconciliation |
| Integration or Engine tree | stack manifest and Engine-tree tests | release construction and reopen verification |
| Release host policy | release, runtime, and host tests | real registry double-build and serving receipts |

When a schema changes, add both a positive construction and a negative reopen
case. Optima's security properties depend heavily on old or malformed objects
failing closed rather than being silently upgraded.

## GPU and OCI work

Use the exact SGLang pin and deployment-approved Torch/CUDA combination from
[GPU setup](gpu-setup.md). Run `python -m optima.cli compat`. The version row
compares the installed package with `PINNED_SGLANG`; a mismatch is marked
`DIFFERS from pin`, fails the row, and makes the command exit nonzero.

Production-shaped candidate tests need more than Docker access. They need the
reviewed no-egress image, non-root worker policy, bounded mounts, exact model
and native publications, device cleanup, host timing, and an injected arena
provider. Do not approximate that trust boundary with an arbitrary container
command and call the result production qualification.

## Before opening a change

- Run the focused tests for every contract you touched.
- Run the complete CPU-eligible suite and account for every skip.
- Re-run the SGLang and chain canaries when their adapters changed.
- Update the relevant guide and reference page with the executable contract.
- Record hardware, runtime, model, topology, graphs, and workload identity for
  any performance claim.
- State explicitly whether evidence is diagnostic, historical, structural, or
  produced by the current qualification path.

## Source layout

See the [codebase map](../reference/codebase-map.md) for subsystem ownership.
The executable package is `optima/`, validator-owned reference kernels are in
`optima_kernels/`, examples are in `examples/`, and contract tests are in
`tests/`.

Source: [`pyproject.toml`](https://github.com/latent-to/cacheon/blob/main/pyproject.toml).
