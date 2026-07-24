# Bundle format

A component bundle is a source-only proposal for one registered contribution
target. Its `manifest.toml` is data: the validator parses it before importing
miner code.

The parser is implemented in
[manifest.py](https://github.com/latent-to/cacheon/blob/main/optima/manifest.py).
Target resolution is a separate step implemented by
[target_catalog.py](https://github.com/latent-to/cacheon/blob/main/optima/target_catalog.py).
A manifest can be structurally valid and still be ineligible for a registered
target.

## Recommended layout

```text
my_bundle/
  manifest.toml
  kernels/
    kernel.py
  metadata/
    kernel.json
  README.md
  LICENSE
```

Advanced, target-approved bundles may also contain declared `.cu`/`.cuh`
sources, unified dependency diffs, a reviewed `rebuild.json`, or a CuTe source
module with a sealed direct-artifact declaration. Do not include compiled
objects, CUBINs, wheels, shared libraries, caches, model weights, credentials, or
absolute machine paths.

All declared paths are bundle-relative. Intake rejects path traversal, unsafe
filesystem entries, and undeclared executable material. The fetched archive is
rehash-checked and republished as immutable validator-owned input before any
screen or qualification runs.

### What each file is for

| File | Identity and execution role |
|---|---|
| `manifest.toml` | names the proposal ABI, requested target, implementation rows, and paths; it is parsed as data before candidate import |
| `kernels/*.py` | candidate source loaded only after structural and static-policy gates, and only inside the appropriate worker boundary |
| `metadata/*.json` | routing eligibility such as graph declaration, model, phase, dtype, topology, and shape domain; it cannot set correctness tolerances or reward policy |
| declared `.cu`/`.cuh` | inspectable native inputs to a validator-approved rebuild step; never prebuilt output |
| declared `.patch`/`.diff` | exact source delta for the narrowly allowed pinned dependency surface |
| `rebuild.json` | selects registered validator-owned build capabilities; it is not a shell script |
| `README*`, `LICENSE*` | human context and licensing; still identity-bearing bytes, so changing them creates a different content hash |

“Source-only” does not mean “only `.py`.” It means that every executable product can be
derived by a registered builder from visible, identity-bound inputs. A `.so`, wheel,
object file, cache entry, or undeclared source cannot be made acceptable by mentioning it
in a README.

### Three validations happen at different times

1. **Manifest validation** checks TOML structure, identifiers, safe relative paths, and
   declaration shapes. It does not import the entry or decide target eligibility.
2. **Tree/static validation** inventories the recursive tree and scans every Python
   source. It rejects obvious escape/egress patterns and undeclared executable material.
   This is defense in depth, not a sandbox.
3. **Target resolution and execution validation** compare independently observed bundle
   features with the active target catalog, then build and exercise the selected delta
   in validator-owned isolation.

A bundle can pass the first two and fail the third. For example, `setup = "initialize"`
is valid manifest syntax, but no current registered target permits that observed feature.

## A competitive singleton bundle

Use an explicit competition table even though old development examples may
still resolve through a legacy singleton convenience:

```toml
bundle_id = "alice-silu-sm90-v1"
abi_version = "optima-op-abi-v0"

[competition]
target = "activation.silu_and_mul"
mode = "slot"

[[ops]]
slot = "activation.silu_and_mul"
source = "kernels/silu_and_mul.py"
entry = "silu_and_mul"
dtypes = ["bfloat16", "float16"]
architectures = ["sm90"]
metadata = "metadata/silu.json"
```

`competition.target` selects a validator-registered reward unit. It does not
create a target or alter its contract.

### Complete anatomy of this example

The matching source file must export the named callable and write the supplied output:

```python
# kernels/silu_and_mul.py
import torch


def silu_and_mul(x, out):
    d = x.shape[-1] // 2
    gate = torch.nn.functional.silu(x[..., :d].float()).to(x.dtype)
    out.copy_(gate * x[..., d:])
```

The metadata narrows where the row may route:

```json
{
  "graph_safe": true,
  "capabilities": {
    "num_tokens": {"min": 1, "max": 4096}
  }
}
```

Read the three files together:

- the competition table asks for the economic target;
- the op row maps one semantic slot to one source callable;
- manifest and metadata constraints intersect to form the effective routing domain;
- the source implements the computation but does not choose when it is called; and
- the target catalog supplies correctness, reference, feature permissions, overlap, and
  serving binding.

The numeric range is a routing claim, not a verifier-shape request. If a live call
descriptor lacks a field you constrain, routing fails closed rather than treating the
field as a wildcard.

Valid modes are:

- `slot` for a registered singleton target;
- `atomic` for a registered multi-slot target;
- legacy `system`, which remains parseable for migration but is never
  registered or crownable.

## Op rows

Each `[[ops]]` row describes one implementation:

| Field | Meaning |
|---|---|
| `slot` | semantic slot implemented by this row |
| `source` | bundle-relative Python source module |
| `entry` | callable exported by that module |
| `variant` | explicit implementation identity; required on every row when a slot has more than one row |
| `dtypes` | allowed runtime dtypes |
| `architectures` | allowed canonical GPU architectures such as `sm90` or `sm103` |
| `metadata` | bundle-relative JSON eligibility metadata |
| `prepare` | optional load-time weight preparation callable for a slot whose ABI permits it |
| `setup` | engine-wide setup hook; currently forbidden by every registered target |
| `base_kernel`, `override_point` | registered override composition fields |
| `cuda_sources` | declared inspectable `.cu`/`.cuh` inputs to an approved build step |
| `aot_exports` | sealed compiler exports, semantic bindings, specializations, lifecycle steps, and complete device plans |
| `artifact_resources` | miner-named but validator-allocated workspace, prepared storage, or engine state |

`bundle_id` must be a simple non-empty identifier, and the component ABI is
currently exactly `optima-op-abi-v0`.

Unknown op keys are preserved as extra data, but that does not make them
meaningful or allowed by target policy.

## Capability metadata

The validator routes a variant only when its entire declared domain matches a
validator-produced live call descriptor. Missing fields are mismatches, not
wildcards.

Use the `capabilities` object for named specialization predicates:

```json
{
  "graph_safe": true,
  "capabilities": {
    "model": {"exact": "MiniMax-M3-NVFP4"},
    "phase": {"exact": "prefill"},
    "head_dim": {"exact": 128},
    "block_size": {"one_of": [64, 128]},
    "q_len": {"min": 256, "max": 8192},
    "tp_size": {"exact": 4}
  }
}
```

A scalar is shorthand for `exact`, and an array is shorthand for `one_of`.
Numeric fields may use inclusive `min`/`max`. Supported fields currently include:

- context: `architecture`, `dtype`, `graph_mode`, `layout`, `model`, `phase`,
  `quant`, `runtime`;
- dimensions/topology: `alignment`, `batch_size`, `block_size`, `ep_size`,
  `exp_tokens`, `head_dim`, `hidden_dim`, `intermediate_dim`, `kv_len`,
  `last_dim`, `num_experts`, `num_kv_heads`, `num_q_heads`, `num_tokens`,
  `page_size`, `q_len`, `top_k`, `tp_size`, `world_size`.

Unknown capability names fail at metadata load. Manifest `dtypes` and
`architectures` are real routing constraints and intersect with metadata
constraints. Legacy metadata keys such as `min_num_tokens`, `max_num_tokens`,
`max_last_dim`, `quant`, and `graph_safe` are still interpreted, but new
specializations should be expressed in `capabilities` where possible.

The canonical vocabulary and normalization rules are in
[capabilities.py](https://github.com/latent-to/cacheon/blob/main/optima/capabilities.py)
and routing is in
[registry.py](https://github.com/latent-to/cacheon/blob/main/optima/registry.py).

An empty manifest list contributes no constraint for that field; a non-empty list does.
If both manifest and metadata declare a field, their canonicalized intersection is the
domain. A disjoint intersection is invalid, which catches mistakes such as BF16 in TOML
and FP16-only metadata before candidate execution.

## Multiple variants

Multiple implementations of one slot are permitted only when all rows name
unique variants and their effective capability domains are provably disjoint.
Manifest order never chooses a winner.

```toml
[[ops]]
slot = "activation.silu_and_mul"
variant = "small"
source = "kernels/small.py"
entry = "silu_small"
metadata = "metadata/small.json"

[[ops]]
slot = "activation.silu_and_mul"
variant = "large"
source = "kernels/large.py"
entry = "silu_large"
metadata = "metadata/large.json"
```

For example, `small.json` might cap `num_tokens` at 127 while `large.json`
starts at 128. If overlap exists—or cannot be resolved safely—the registry
rejects the bundle rather than inventing priority.

This matters because variant selection is a semantic decision, not a source-order
convenience. If two rows match the same call, the validator cannot know which selected
delta was measured or which implementation would later serve. Conversely, a gap is
allowed: the trusted incumbent remains the fallback outside all candidate domains.

## Direct CUBIN bundles

The registered direct-artifact provider is `cutlass.cute.cubin.v1`. It compiles
CuTe source into a sealed CUBIN during validator prebuild and executes it through
the validator's CUDA Driver ABI. It does not run a miner-supplied launcher in the
scheduler process.

A direct bundle contains source and declarations, not generated native output:

```text
my_direct_bundle/
  manifest.toml
  kernels/
    kernel_cute.py
  metadata/
    kernel.json
  rebuild.json
  README.md
  LICENSE
```

The resolved rebuild selection is data, not a command:

```json
{
  "steps": [
    {"type": "repo_python", "path": "build_cute_cubin.py"}
  ]
}
```

The operation row retains `entry` because the outer bundle ABI requires it. For
direct execution the value is not called and is excluded from canonical direct-
execution identity. The meaningful declaration is `[[ops.aot_exports]]`, which
must include the provider, compiler-side factory, bounded compile-profile inputs,
ordered bindings, specialization/lifecycle fields, and a complete
`optima.device-launch-plan.v1` plan.

The plan is intentionally explicit. It inventories every logical kernel and
formal parameter width, then declares every launch's grid, block, optional
cluster, shared memory, parameter construction, stream, and allowlisted
attributes. Parameter values are constructed by validator code from checked
expressions and admitted slot resources. Supported constructions include exact
scalars and pointers, packed structs, TMA descriptors, CuTe/CUTLASS FastDivmod,
and provider-authorized group projections.

If the algorithm needs intermediate storage, declare
`[[ops.artifact_resources]]`. `workspace.*` is call-local, `prepared.*` crosses
the validator's prepare/run boundary, and `state.*` persists with the engine
entry. The validator allocates all three. Resource declarations cannot create a
new stream, group, semantic input, output, or arbitrary address.

Direct-artifact admission is exact:

- the target must permit `aot:cutlass.cute.cubin.v1` and
  `rebuild:build_cute_cubin`;
- the factory runs only in the no-network/no-GPU compiler child with declared
  validator-measured profile values;
- the only executable products are CUBINs; the sealed provider publication also carries
  its canonical index, while the enclosing native publication carries its prebuild and
  inventory metadata, all bound to build/profile/tree identities;
- the scheduler admits the complete driver-observed CUBIN ABI by kernel ordinal
  after rank CUDA setup; and
- qualification requires load, invocation, and seam-completion evidence on every
  active member, with no fallback.

The provider vocabulary includes native group handles and peer-pointer tables, but
the standard `build_cute_cubin` load path does not install the required group
capability/handle resolvers; such a plan fails closed before execution. Group-aware
support therefore requires reviewed runtime work before the ordinary distributed
correctness, graph, full-engine, and reproduction gates can produce evidence.

Use the field tables in [Bundle manifest](../reference/manifest-schema.md) and the
runtime model in [Sealed direct artifacts](../architecture/direct-artifacts.md).

## What the content hash binds

`chain-package` computes proposal identity from every regular identity-bearing file in
sorted relative-path order, with paths and contents length-prefixed before SHA-256. It
then puts exactly those files under one archive wrapper directory. Filesystem modes,
gzip compression details, `.git`, `__pycache__`, `.pyc`, `.pyo`, and AppleDouble `._*`
noise do not define the content hash; production extraction rejects identity-excluded
archive state rather than placing unhashed material next to the proposal.

Practical consequences:

- renaming a source file changes the hash even if its bytes do not;
- editing metadata, README, license, or the manifest changes the hash;
- recompressing the same canonical tree need not change the content hash;
- editing the local tree after packaging makes `chain-submit` commit a different hash
  from the hosted archive; and
- a fix after reveal is a new content identity and therefore a new proposal, not a
  mutation of the old one.

The archive URL is transport, while the content hash is proposal identity. The validator
may retry transient transport for the same hash, but it must never accept different
extracted bytes under that identity.

For a direct row, a second canonical identity covers the executable declaration:
provider, factory, profile inputs, bindings, resources, lifecycle, specialization,
prelaunch, and the complete device plan. Copy/provenance analysis combines that
declaration with the normalized transitive Python source closure. Reformatting or
renaming an unused `entry` does not change direct execution, while any
execution-bearing declaration change does.

## Prepare/forward slots

MoE slots use a load-time preparation callable plus a serving callable:

```toml
[competition]
target = "moe.fused_experts"
mode = "slot"

[[ops]]
slot = "moe.fused_experts"
source = "kernels/moe.py"
prepare = "prepare"
entry = "fused_experts"
```

`prepare(w13, w2)` may derive a weight layout. The serving call receives the
returned object as `prepared`. This does not authorize engine-wide mutation.

## Atomic bundles

An atomic bundle must request a registered atomic target and implement its
complete member set. The current example shape is:

```toml
[competition]
target = "collective.moe_epilogue.v1"
mode = "atomic"

[[ops]]
slot = "collective.ar_residual_rmsnorm"
source = "kernels/epilogue.py"
entry = "ar_residual_rmsnorm"

[[ops]]
slot = "collective.moe_finalize_ar_rmsnorm"
source = "kernels/epilogue.py"
entry = "moe_finalize_ar_rmsnorm"
```

The committed
[atomic stack fixture](https://github.com/latent-to/cacheon/tree/main/tests/fixtures/stack_fused_epilogue_atomic)
is useful for studying identity and rebuild declarations. It is a test fixture,
not a production-ready kernel and not a template whose stub ABI should be
copied blindly.

## Advanced declarations

These fields are valid only where the registered target explicitly allows the
corresponding observed feature:

```toml
[[dep_patches]]
target = "flashinfer"
path = "patches/fused_moe.patch"
```

```toml
[[ops]]
slot = "moe.fused_experts"
source = "kernels/epilogue.py"
entry = "gemm1_epilogue"
base_kernel = "nvfp4_moe_megakernel"
override_point = "gemm1_epilogue"
cuda_sources = ["kernels/epilogue_sm103.cu"]
```

Declarations do not bypass policy. Intake independently observes rebuild,
override, CUDA-source, and dependency-patch features and resolves them against
the target catalog. See [Override points](override-points.md) and
[Dependency patches](dep-patches.md).

Direct-artifact exports follow the same rule: the closed provider registry and
target catalog, not a provider string in TOML, determine whether a row is
authoritative and crownable.

Next, read the [Kernel ABI](kernel-abi.md), then build the minimal example in
[Your first kernel](your-first-kernel.md).
