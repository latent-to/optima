# Bundle manifest

Every contribution bundle has a `manifest.toml`. The manifest is parsed as
data before any contribution module is loaded. Its current ABI identifier is
`optima-op-abi-v0`.

## What parsing does—and does not do

A manifest passes through distinct layers:

```mermaid
flowchart LR
    T["TOML + contained paths"] --> M["Manifest object"]
    M --> F["Observed bundle features"]
    F --> C["Target-catalog resolution"]
    C --> S["Static scan"]
    S --> V["Component verification"]
    V --> Q["Arena qualification"]
```

`load_manifest()` covers only the first step: required syntax, identifier
shape, path containment/existence, variant uniqueness, and structural CUDA or
patch declarations. It does not import source, prove that the requested target
exists, or verify numerical behavior. `scan`, target resolution, `verify`, and
production qualification are separate gates so a structurally valid document
cannot grant itself authority.

## Minimal singleton example

```toml
bundle_id = "example-silu-v1"
abi_version = "optima-op-abi-v0"

[competition]
target = "activation.silu_and_mul"
mode = "slot"

[[ops]]
slot = "activation.silu_and_mul"
source = "kernel.py"
entry = "silu_and_mul"
dtypes = ["float32"]
architectures = ["cpu"]
```

`bundle_id` and identifiers accept letters, numbers, `.`, `_`, and `-`.
Paths are relative to the bundle and must resolve to regular contained files.

## Top-level fields

| Field | Required | Meaning |
|---|---:|---|
| `bundle_id` | yes | Human-readable bundle identifier; not the content identity |
| `abi_version` | yes | Must equal `optima-op-abi-v0` |
| `[competition]` | recommended | Explicit requested target and `slot`/`atomic` mode |
| `[[ops]]` | yes | One or more implementation rows |
| `[[dep_patches]]` | no | Declared text patches for a validator-approved dependency lane |

The canonical bundle hash, not `bundle_id`, is the proposal's content identity.

### Identity versus display name

The content hash walks the bundle's own regular files in sorted relative-path
order and hashes length-prefixed path and byte sequences. Git and Python cache
noise is excluded; symlinks are not part of the hashed file set and the bundle
loader/scanner rejects unsafe tree structure. Editing source, metadata, a
patch, or even the manifest produces a new identity. Renaming only the outer
directory does not.

`bundle_id` is useful in logs and diagnostics but never proves that two reveals
contain the same artifact. Commit/reveal, immutable publication, copy checks,
and qualification use digest-bound content.

### Competition table

| Field | Values |
|---|---|
| `target` | A validator-registered target ID |
| `mode` | `slot` or `atomic`; `system` parses only for legacy migration |

The table is a request, not policy. Intake resolves it against the frozen
[target catalog](target-catalog.md) and complete observed feature set. The
current resolver can infer a target when an exact singleton or registered
atomic member set is unambiguous, which preserves older bundles. New
competitive bundles should declare `[competition]` explicitly.

## Operation rows

| Field | Required | Meaning |
|---|---:|---|
| `slot` | yes | Registered execution slot |
| `source` | yes | Python source module within the bundle |
| `entry` | yes | Entry callable name |
| `variant` | conditional | Capability variant; required on every row when a slot repeats |
| `prepare` | no | Weight-preparation callable for a registered prepare/forward ABI |
| `setup` | development only | Legacy parser/direct-framework diagnostic hook; every registered component target forbids it |
| `dtypes` | no | Declared dtype capability domain; an empty array adds no dtype restriction |
| `architectures` | no | Declared architecture capability domain; an empty array adds no architecture restriction |
| `metadata` | no | Eligibility/capability JSON within the bundle |
| `base_kernel` | no | Validator-owned base-kernel identifier for an override submission |
| `override_point` | no | Typed hole in that base; requires `base_kernel` |
| `cuda_sources` | no | Inspectable `.cu`/`.cuh` inputs for a reviewed builder |
| `aot_exports` | no | Ordered sealed direct-artifact exports for a slot with a declarative call ABI |
| `artifact_resources` | no | Validator-allocated storage shared by the row's artifact lifecycle |

Unknown operation fields are retained for observation. They do not grant a
capability; complete feature resolution can reject a bundle that asks for more
than its target admits.

### How an operation row is selected

The `slot` chooses a validator-owned semantic ABI, not an arbitrary import
hook. The validator resolves `source` inside the bundle and looks up the named
Python identifier only after structural and static gates. It allocates outputs
and passes arguments in the registered slot order. Candidate code fills those
outputs; it does not redefine shapes, references, tolerances, or the call site.

For prepare/forward slots, `prepare` names the registered one-time weight
transformation while `entry` names the runtime call. `setup` is a legacy direct
framework hook and every current component target forbids it, so new
competitive manifests should not use it.

For a row with `aot_exports`, `source` is scanned and its declared compiler
factory is imported only in the isolated prebuild compiler child. The scheduler
does not import that source as a runtime launcher. `entry` remains required by
the outer `optima-op-abi-v0` syntax, but direct execution canonicalizes it to the
validator entry `_optima_direct_artifact`; changing the unused value does not
change direct-execution identity.

### Variants

Several rows may implement the same semantic slot only when every row has a
unique explicit `variant`. Their validator-parsed capability domains must not
overlap. Runtime selection is fail-closed: exactly one applicable variant is
required, otherwise the trusted baseline is used or qualification refuses the
candidate according to the registered boundary.

Empty or omitted `dtypes` and `architectures` do not mean “supports nothing”;
they add no restriction at the manifest-parser layer. Eligibility metadata and
validator observation still constrain the real capability domain. Conversely,
listing `sm103` does not prove the source builds or runs there.

```toml
[[ops]]
slot = "norm.rmsnorm"
variant = "sm90-bf16"
source = "rms_sm90.py"
entry = "rmsnorm"
dtypes = ["bfloat16"]
architectures = ["sm90"]

[[ops]]
slot = "norm.rmsnorm"
variant = "sm103-bf16"
source = "rms_sm103.py"
entry = "rmsnorm"
dtypes = ["bfloat16"]
architectures = ["sm103"]
```

## Sealed direct-artifact exports

`[[ops.aot_exports]]` is a strict, closed table. It is accepted only for a slot
that has an immutable validator-defined artifact call ABI and only when target
resolution permits the observed provider and rebuild features.

| Field | Required | Contract |
|---|---:|---|
| `provider` | yes | Registered provider ID; the registry contains `cutlass.cute.cubin.v1` |
| `name` | yes | Export-local canonical identifier |
| `factory` | yes | Identifier called only in the no-egress compiler child |
| `profile_inputs` | yes | Unique list drawn from the provider's compile-profile allowlist |
| `bindings` | yes | Ordered projections of immutable slot resources and declared artifact resources |
| `device_plan` | yes for the registered provider | Complete `optima.device-launch-plan.v1` declaration |
| `role` | no | `init`, `prepare`, `reset`, `run`, or `destroy`; default `run` |
| `plan` | no | Specialization-plan name; default `default` |
| `step` | no | Ordered non-negative step within the plan; default `0` |
| `specializes` | no | Exact-equality predicates over call-ABI resources |
| `prelaunch` | no | Bounded validator operation list; the registered operation is exact `fill` |
| `provider_capability_requirements` | sealed/reopen | Canonical requirements derived from binding projections |
| `specialization_capability_requirements` | sealed/reopen | Canonical requirements derived from specialization sources |

The provider compile-profile allowlist is
`max_active_clusters.cluster_size_{1,2,4,8,16}`. A factory can receive only the
keys named in its `profile_inputs`; it cannot query a GPU in prebuild.

TOML row order is not executable authority. The parser canonicalizes exports by
provider, plan, step, and name. Within each plan, steps must be unique and their
step-sorted lifecycle roles must follow `init`, `prepare`, `reset`, `run`,
`destroy`; every step in that plan carries the same specialization predicate.
Plan overlap is legal only as a strict fallback chain. Runtime selects the unique
most-specific exact match rather than using source order.

### Binding rows

Each binding row has a `kind` and, except for `aggregate`, a canonical `source`
such as `input.q`, `output.out`, `stream.current`, or a declared
`workspace.*` resource. Available projections are a closed relation to the
source kind:

| Binding kind | Main projections and options |
|---|---|
| `tensor` | `descriptor`; optional bounded `unsqueeze`, checked `assumed_align`, and `leading_dim` |
| `pointer` | `device_ptr` from tensor storage, `identity` from a pointer, or provider-authorized `native_handle` / `peer_ptr_table` from a group |
| `scalar` | `value`, or tensor/group metadata `shape`, `stride`, `numel`, `rank`, `size`, `storage_offset`; requires an exact scalar `cast` |
| `stream` | `identity` for the validator's captured stream |
| `group` | `identity` for the supplied semantic group |
| `opaque` | `identity` for a validator-held opaque slot resource |
| `aggregate` | Bounded nested `Shape`, `Coord`, `Tile`, `IntTuple`, or `Stride` value with `i32`/`i64` dynamic leaves |

An axis is required for `shape` and `stride`. A `peer_ptr_table` binding must
name the exact persistent `group_ipc` artifact resource in `peer_resource`.
Bindings cannot name a callback, allocate a new stream or group, or construct an
integer address.

### Artifact resources

Each `[[ops.artifact_resources]]` row has exactly these authored fields:

| Field | Required | Contract |
|---|---:|---|
| `name` | yes | `workspace.*`, `prepared.*`, or `state.*` |
| `dtype` | yes | Registered storage dtype |
| `alignment` | yes | Power of two, at least dtype width and at most 1 MiB |
| `lifetime` | yes | Fixed by the name prefix: `call`, `prepared`, or `engine` |
| `shape` | yes | One to 16 bounded extent tables |
| `scope` | no | `rank_local` by default or persistent `group_ipc` |

Each shape extent contains `factors` and optional `divisor`. A factor is either
`{ static = N }` or a dynamic table with `source`, `projection`, `upper_bound`,
and `axis` when projecting a tensor dimension. The extent is
`ceil(product(factors) / divisor)`. This is the entire allocation-expression
language. Dynamic sources must be resources in the immutable slot call ABI, not
other artifact buffers. A prepared-lifetime shape may use only resources present
at the validator's exact prepare-allocation boundary.

Prefix, lifetime, and role are validated together:

| Prefix | Lifetime | Lifecycle rule |
|---|---|---|
| `workspace.*` | `call` | Cannot be used by `init` or `destroy` |
| `prepared.*` | `prepared` | Must be produced in `prepare` and consumed in `run`; cannot be used by `init` or `reset` |
| `state.*` | `engine` | Persists with the artifact entry and is available to registered lifecycle roles |

`group_ipc` is invalid for call-local workspace. Manifest ceilings are 32
resources, 64 GiB per resource, and 128 GiB aggregate; runtimes may impose lower
live limits. These declarations request validator-owned storage, not candidate
allocation authority.

### Device launch plan

The registered provider requires `device_plan.schema =
"optima.device-launch-plan.v1"` plus exact `kernels` and `launches` inventories.

| Inventory | Required contents |
|---|---|
| `kernels` | Sorted unique logical `name` and complete `parameter_sizes` byte vector |
| `launches` | Contiguous `ordinal`, logical `kernel`, `grid`, `block`, optional `cluster`, `shared_mem_bytes`, ordered `parameters`, `stream_binding`, and allowlisted `attributes` |

Grid, block, cluster, shared memory, and scalar parameter values are checked
expression trees over admitted live bindings. Parameter kinds are `pointer`,
`scalar`, `packed_struct`, `tma_descriptor`,
`cutlass_fast_divmod_i32_v1`, `cute_fast_divmod_i32_v1`, and
`group_handle`. The parameter-size vector of every launch must exactly match the
logical kernel contract.

The plan schema is bounded to 1,024 logical kernels, 256 launches, and 64 semantic
bindings. Dynamic shared memory is expression-derived; the runtime evaluates it
for each invocation and rejects values above 1 MiB. At runtime, canonical logical
kernel ordinal is matched to the complete driver-observed physical CUBIN inventory
and exact per-ordinal widths. Candidate-provided physical-symbol search is not
part of the ABI.

See [Sealed direct artifacts](../architecture/direct-artifacts.md) for build,
admission, evidence, identity, and support boundaries.

## Dependency patches

```toml
[[dep_patches]]
target = "flashinfer"
path = "patches/change.patch"
```

Only UTF-8 `.patch`/`.diff` files are structurally accepted. Binary, rename,
and deletion patches are refused. Admission still requires a target that
permits the observed dependency-patch feature and a validator-reviewed applier
with a bounded destination policy.

CUDA source declarations behave similarly. Listing `.cu`/`.cuh` paths makes
them inspectable inputs to the sanctioned build lane; it is not permission to
ship a prebuilt binary or execute an arbitrary compiler command. Rebuild
operations come from reviewed validator policy.

## What the manifest cannot do

A bundle cannot choose:

- its reward family, overlap, or displacement;
- arena hardware, workload, thresholds, or hidden tasks;
- the incumbent, reference, or release stack;
- isolation or network policy;
- qualification, reproduction, or settlement outcomes.

## Failure diagnosis

| Error class | Typical cause | Fix the right layer |
|---|---|---|
| TOML/required-field error | Missing `[[ops]]`, wrong ABI string, invalid identifier | Correct manifest syntax |
| Path error | Absolute path, `..` escape, missing file, unsafe symlink | Keep every declared input as a regular contained file |
| Duplicate-slot error | Multiple rows without explicit unique variants | Name every variant and make domains disjoint |
| Competition error | Unknown target, wrong `slot`/`atomic` mode, legacy `system` title | Choose a registered target from validator output |
| Feature-admission error | CUDA, patch, override, setup, or extra capability outside target policy | Remove the feature or choose the registered lane |
| Artifact-provider error | Unknown provider, missing rebuild feature, or non-crownable provider | Use the registered provider only on a target that admits it |
| Artifact-ABI error | Binding source/projection, resource lifecycle, or device parameter widths disagree | Reconstruct the declaration from the slot call ABI and complete CUBIN contract |
| Compile-profile error | Factory requests an undeclared constant or architecture/profile authority differs | Declare only allowlisted inputs and rebuild for the exact arena profile |
| Static scan error | Forbidden import/operation or uninspectable tree content | Rewrite the source; scanning is not a sandbox exception list |
| Verification error | Callable/signature/output/reference mismatch | Debug the registered tensor and correctness contract |
| Qualification failure | Complete engine misses timing, drift, quality, fidelity, or resource gates | Inspect retained arena evidence; do not relabel the outcome |

## Pre-submission checklist

- Run `slots` and select the economic target separately from its execution
  slot members.
- Declare `[competition]` explicitly for new singleton or atomic work.
- Keep `bundle_id` descriptive but assume only the content hash is identity.
- Declare every source, metadata, CUDA, and dependency-patch input with a
  contained relative path.
- Give repeated slot rows unique variants with non-overlapping domains.
- Avoid `setup`; use only registered `prepare` and entry contracts.
- For direct artifacts, declare the complete binding, resource, lifecycle, and
  device plan; never package generated CUBIN or a host launcher.
- Run `scan`, then the appropriate local or collective `verify` command.
- Hash and package the exact verified tree; any later byte change is a new
  proposal.

Source: [`optima/manifest.py`](https://github.com/latent-to/cacheon/blob/main/optima/manifest.py).
