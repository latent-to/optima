# Local quickstart

This quickstart exercises bundle parsing, static policy, and slot-level correctness on
CPU. It does not reproduce production qualification and cannot create a crown.

## Prerequisites

- Python 3.10 or newer
- Git
- enough local storage for a CPU PyTorch installation

Clone the source and create an isolated environment:

```bash
git clone https://github.com/latent-to/cacheon.git
cd cacheon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cpu,dev,release]"
```

## Inspect the slot catalog

```bash
python -m optima.cli slots
```

The command prints the registered slot names, kinds, tensor semantics, and callable
signatures. Economic targets are a separate validator-owned layer; see the
[target catalog](../reference/target-catalog.md).

Read one row as a contract rather than a menu of functions. For example,
`activation.silu_and_mul` tells you that the validator supplies one input and one output,
your callable fills the output, and the trusted reference defines the result. It does
not say that a miner may replace the surrounding MLP, choose output storage, or declare a
new reward category.

## Scan without executing code

```bash
python -m optima.cli scan examples/miner_silu_torch
```

`scan` parses the manifest and applies the recursive static policy. A clean scan is an
admission signal, not a sandbox and not correctness evidence.

Expected output for the clean example:

```text
bundle: example-silu-torch-cpu  abi: optima-op-abi-v0  ops: 1
  [clean] activation.silu_and_mul <- kernels/silu_and_mul.py
```

Interpret each part precisely:

- `bundle` and `abi` came from `manifest.toml`;
- `ops: 1` means one declared implementation row, not one registered target;
- `[clean]` means the declared source passed static policy; it does not mean the callable
  was imported, numerically tested, graph-captured, or accepted by production target
  resolution.

If `scan` reports `VIOLATIONS`, fix the named source or recursive-tree finding instead of
trying `verify`. Typical causes are forbidden file/process/network APIs, an undeclared
native file, a compiled artifact, or executable Python vendored outside the declared
entry. A traceback before this summary usually means the manifest itself could not be
parsed or a declared path was missing or unsafe.

## Verify a faithful implementation

```bash
python -m optima.cli verify examples/miner_silu_torch \
  --device cpu --dtype float32
```

The validator allocates inputs and outputs, invokes the candidate through the slot ABI,
and compares it with the trusted reference.

On CPU, expect a note that the run checks op correctness only, followed by a variant
summary and one row per exercised shape. This example declares graph-safe operation, so
the CPU headline is `NUMERICAL_PASS ... graph=NOT_VERIFIED`; its individual numerical
shape rows are `ok`, but CPU cannot supply the required CUDA replay evidence. Read the
fields this way:

| Output | Interpretation |
|---|---|
| `PASS` | every applicable diagnostic shape passed its numerical contract |
| `FAIL` | at least one applicable shape failed, or the variant/domain preflight was invalid |
| `N/A` | this variant does not apply to the selected invariant context; no candidate code was invoked for it |
| `NUMERICAL_PASS` | numerical checks passed but required CUDA-graph proof did not complete |
| `max_abs`, `max_rel` | worst reported errors; the target's comparator, not either number alone, decides pass/fail |
| `ratio`, `cos`, `overlap` | the active semantic metric for tolerant, low-bit, or MSA selection contracts |
| `graph_replays` | successful checked replays; absent on this CPU tutorial |

Individual shape rows can be N/A when a declared domain excludes them. If every bundle
variant is context-inapplicable, `verify` exits nonzero: “nothing ran” is not correctness
evidence. Likewise, a CPU `PASS` is not a hidden GPU pass.

Now run the adversarial example:

```bash
python -m optima.cli verify examples/miner_silu_broken_torch \
  --device cpu --dtype float32
```

The broken implementation drops the SiLU operation. It is cheaper work, but it must fail
correctness.

The command should exit nonzero and identify failed shapes. That is a useful control
experiment: it proves this checkout is loading the requested bundle and that the
reference comparison is active. Do not diagnose a failure from error magnitude alone.
Check the shape, comparator/detail field, output completeness, input mutation, callable
signature, and declared domain in that order. The miner
[diagnostics guide](../miner-guide/diagnostics.md) maps the same reasoning to GPU, graph,
transport, and production lifecycle failures.

## Run the test suite

```bash
pytest -q
```

The `release` extra supplies the cryptographic dependency used by the complete
release tests. The suite covers much more than the CPU tutorial: manifests, target resolution, stack
assembly, hostile transport, OCI policy, qualification evidence, settlement, emissions,
release construction, and compatibility guards. GPU-specific tests skip when their
runtime is unavailable.

## What this did not prove

The quickstart did not exercise:

- SGLang model execution;
- distributed collective verification;
- CUDA graph capture and replay;
- no-egress OCI candidate execution;
- v3 resident B/C/B′, conditional C′/B″, registered eager audit A, and
  pristine T qualification;
- independent reproduction or settlement; or
- release construction and serving.

Those boundaries require validator-owned hardware, runtime identities, policies, and
evidence stores. Developer GPU experiments remain non-authoritative regardless of their
local output.

## Continue

| If you want to… | Read… |
|---|---|
| write your first bundle | [Your first kernel](../miner-guide/your-first-kernel.md) |
| prepare a GPU development machine | [GPU setup](../dev/gpu-setup.md) |
| understand production qualification | [Qualification](../validator-guide/qualification.md) |
| operate finalized intake first | [Deployment readiness](../validator-guide/first-hour.md) |
