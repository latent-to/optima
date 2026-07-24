# Kernel ABI

The slot ABI is a write-into-output contract. The validator owns the call site,
input bindings, output allocation, reference behavior, and downstream engine.
Your implementation owns only the declared computation.

## Why the ABI is shaped this way

An inference runtime already owns long-lived tensor storage, graph-captured addresses,
streams, process groups, and downstream consumers. Allowing a candidate to return an
arbitrary replacement tensor would let it silently change allocation, aliasing, layout,
device, synchronization, and lifetime along with the math. That would make the measured
delta wider than the registered target and make graph replay unreliable.

Write-into-output keeps the ownership line observable:

```text
validator                         candidate                     validator
---------                         ---------                     ---------
allocate/fill inputs  ------->    read inputs
allocate + poison out ------->    write all logical cells  ---> validate binding
supply scalar/group    ------->    perform slot semantics   ---> compare reference
retain downstream path <------------------------------------    consume same storage
```

Poisoning is important. If the validator fills `out` with NaNs or sentinel data before a
call, a partial write cannot accidentally pass because an old buffer still contains
plausible values. Replaying with fresh logical inputs while keeping captured addresses
stable also detects kernels that bake capture-time data into the graph.

The ABI is therefore both a programming interface and the boundary of the causal claim.
You are free to choose algorithms, tiling, fusion *inside* the slot, and honest
specializations. You do not gain ownership of allocation or adjacent engine semantics.

## Core rules

Every entry implementation must:

- accept the arguments in the slot's exact order;
- write every element of every supplied output;
- honor the supplied output's shape, dtype, device, and stride;
- leave all inputs unchanged;
- remain inside its declared capability domain;
- return `None` (a return value is not used as the model output).

Do not allocate and return a replacement tensor. Do not alias outputs to inputs,
retain live tensors across calls, mutate weights, access the sampler, or infer
that outputs are always contiguous. Verification poisons outputs and checks
input mutation, so partial writes and illegal input reuse fail visibly.

Scalars such as `eps`, `sm_scale`, and `block_size` are inputs, not configuration
requests. Likewise, a supplied `group` is the exact distributed scope for the call; do
not construct a new global group or assume that ambient rank variables describe it.

The authoritative ABI objects are in
[slots.py](https://github.com/latent-to/cacheon/blob/main/optima/slots.py), with
output shape/stride checks in
[tensor_spec.py](https://github.com/latent-to/cacheon/blob/main/optima/tensor_spec.py).

## Direct-artifact form of the ABI

A sealed direct artifact implements the same slot semantics without a runtime
Python callable. Its `bindings` list projects the immutable slot call ABI into a
closed native signature. For example, a simple tensor kernel can bind input and
output device pointers plus the validator's current stream:

```toml
bindings = [
  { source = "input.x", kind = "pointer", projection = "device_ptr" },
  { source = "output.out", kind = "pointer", projection = "device_ptr" },
  { source = "stream.current", kind = "stream" },
]
```

The binding order is the index space used by the device plan. It is not
necessarily the CUDA parameter order: one semantic tensor binding can feed a
pointer, dimension expression, stride expression, packed field, and TMA
descriptor. The validator joins every expression and parameter reference back to
the typed binding before prebuild and again when reopening sealed state.

The complete direct ABI has three parts:

| Layer | Declared by the bundle | Owned at runtime by the validator |
|---|---|---|
| Semantic call | Ordered projections of registered slot resources | Live tensors, scalars, stream, group, and output ownership |
| Artifact storage | Bounded `workspace.*`, `prepared.*`, and `state.*` rows | Allocation, address, scope, budget, and lifecycle |
| Device launch | Complete logical kernels, parameter widths, and ordered launch plans | Physical CUBIN admission, parameter packing, TMA/FastDivmod construction, launch, and cleanup |

Device parameters are a closed set: exact scalars, admitted pointers, packed
structs with checked non-overlapping fields, 128-byte TMA descriptors,
CuTe/CUTLASS FastDivmod values, and provider-authorized group handles. Checked
expressions can read live tensor shape, stride, element size, storage offset,
element count, and admitted scalars. They cannot execute candidate callbacks,
construct arbitrary pointer arithmetic, or discover an ambient stream or group.

The manifest declares logical kernel names because CuTe-generated physical names
depend on the materialized module. After rank CUDA setup, the validator observes
the exact sealed CUBIN through the Driver API and binds the canonical logical
inventory to the physical inventory by ordinal. Kernel count and every formal
parameter width must match. The admitted library handle is retained for launch,
so inspection and execution refer to the same loaded object.

Lifecycle roles are ordered `init`, `prepare`, `reset`, `run`, `destroy`.
`workspace.*` is call-local; `prepared.*` must cross prepare to run; `state.*`
persists with the engine artifact entry. A `prelaunch` fill can initialize an
authorized output or artifact resource, but there is no general host prelaunch
callback.

Direct execution still obeys capability routing, output poisoning, graph replay,
reference comparison, full-engine quality, and reproduction. Qualification also
requires `aot_loaded`, `aot_invoked`, and normal `completed` receipts from every
active scheduler member, with no fallback receipt. Loading a CUBIN is not proof
that it provided the measured result.

Group-aware projections are declaratively recognized only through the supplied
slot group and exact persistent `group_ipc` resources. This schema support does
not make the standard CuTe load path executable: it supplies no group
capability/handle resolvers, so these projections fail closed. A reviewed resolver
integration would still need distributed evidence for each concrete topology and
plan.

See [Sealed direct artifacts](../architecture/direct-artifacts.md) and the
[manifest field reference](../reference/manifest-schema.md).

## Op slots

### `activation.silu_and_mul`

```python
def silu_and_mul(x, out):
    # x: (..., 2*d); out: (..., d)
    d = x.shape[-1] // 2
    out.copy_(torch.nn.functional.silu(x[..., :d]) * x[..., d:])
```

The semantic result is `silu(gate) * up`.

### `norm.rmsnorm`

```python
def rmsnorm(x, weight, out, eps):
    x32 = x.float()
    y = x32 * torch.rsqrt(x32.square().mean(dim=-1, keepdim=True) + eps)
    out.copy_((y * weight.float()).to(out.dtype))
```

This is pure RMSNorm. The slot does not grant ownership of a residual add.

## Attention block slots

### `attention.sdpa`

```python
def attention(q, k, v, out, sm_scale, causal):
    # q: (T, Hq, D); k/v: (S, Hkv, D); Hq is divisible by Hkv
    ...
```

The result is scaled dot-product attention with GQA/MQA expansion and the
validator-provided causal flag.

### `attention.decode`

```python
def attention_decode(q, k, v, seq_lens, sm_scale, out):
    # q: (B, Hq, D); k/v: (B, S, Hkv, D); seq_lens: (B,)
    ...
```

Request `i` attends only to the first `seq_lens[i]` cached keys and values.

### `attention.msa_block_score`

```python
def msa_block_score(q, index_k, seq_lens, block_size, out):
    # out: per-request block-max scores
    ...
```

The validator owns top-k block selection and the subsequent attend. Your output
is a score sheet, and correctness is judged through the selected block sets.

### `attention.msa_prefill_block_score`

```python
def msa_prefill_block_score(q, index_k, prefix_len, scale, block_size, out):
    # q: (T, D); index_k: (S, D)
    # out: (T, ceil(S / block_size)), float32, possibly padded row stride
    ...
```

For query row `m`, key `n` is visible only when
`n <= prefix_len + m`. Invisible score cells use negative infinity. The final
block may be ragged. The output contract deliberately exercises a non-overlapping
row-major strided view, so a kernel that assumes contiguous storage is invalid.

## Prepare/forward MoE slots

`prepare` runs at load time and may build the representation consumed by the
serving entry. It must not mutate the raw inputs.

```python
def prepare(w13, w2):
    # w13: (E, 2*I, H), gate then up; w2: (E, H, I)
    return build_layout(w13, w2)

def fused_experts(x, topk_ids, topk_weights, prepared, out):
    # x: (M, H); routing arrays: (M, K); out: (M, H)
    ...
```

`moe.fused_experts` produces the local expert result. The enclosing trusted path
retains ownership of any later collective.

`moe.fused_experts_reduce` owns that trailing reduction and therefore receives a
process group:

```python
def fused_experts_reduce(
    x, topk_ids, topk_weights, prepared, out, group
):
    # Fill out with the sum of local expert results across group.
    ...
```

The validator does not replay a second stock reduce after this slot. That wider
authority is why it is a distributed contract.

The prepare/forward split exists because weight transformation and request-time work have
different lifetimes. Packing fixed expert weights once can be a legitimate optimization;
packing them on every token would distort the serving path. Conversely, `prepare` is not
an engine initializer: it receives only the registered weight inputs and returns the
representation used by this slot. It cannot patch SGLang, allocate unrelated persistent
state, or inspect future requests.

## Collective slots

Collective verification uses separate processes and the actual supplied group.
Do not create an unrelated global process group or assume rank/world size from
ambient environment variables.

### `collective.all_reduce`

```python
def all_reduce(x, out, group):
    tmp = x.clone()
    torch.distributed.all_reduce(tmp, group=group)
    out.copy_(tmp)
```

### `collective.ar_residual_rmsnorm`

```python
def ar_residual_rmsnorm(
    x, residual, weight, eps, out_norm, out_residual, group
):
    # out_residual = sum_group(x) + residual
    # out_norm = rmsnorm(out_residual, weight, eps)
    ...
```

Both outputs must be filled. `x` differs by rank; `residual` and `weight` are
replicated inputs.

### `collective.moe_finalize_ar_rmsnorm`

```python
def moe_finalize_ar_rmsnorm(
    gemm_out,
    row_map,
    scales,
    residual,
    weight,
    eps,
    out_norm,
    out_residual,
    group,
):
    ...
```

This deep boundary performs four operations as one semantic unit:

1. gather pre-finalize GEMM rows using K-major `row_map`;
2. scale and sum the expert contributions;
3. all-reduce the local partials;
4. add the residual and apply RMSNorm.

`gemm_out` has shape `(T_exp*K, H)`, `row_map` has shape `(T_exp*K)`, and
`scales` has shape `(T_exp, K)`. The live batch may be head-trimmed with
`T <= T_exp`. The deep producer export required to reach this seam is governed
by target and [dependency-patch](dep-patches.md) policy.

## Correctness is target-owned

The validator computes trusted references and applies the target contract. The
current catalog uses:

- elementwise tolerance for numerically equivalent op kernels;
- `matched_ratio` for attention, MoE, and collectives whose legitimate reduction
  order can change rounding;
- per-row `topk_overlap` for MSA score sheets, where selected blocks are the
  semantic output.

Tolerance, ratio, overlap, reference, and model binding are not miner-selected
manifest values. Passing local `verify` demonstrates compatibility with its
diagnostic profiles; authoritative qualification also evaluates the candidate
inside the exact engine and against pristine quality evidence.

The comparators reflect the semantic output of each boundary:

- **all-close** asks whether every output cell implements essentially the same numeric
  operation;
- **matched ratio or cosine** permits the bounded rounding/reduction effects expected of
  a low-bit or reordered implementation without allowing the miner to choose its own
  tolerance; and
- **top-k overlap** grades which blocks the score sheet causes trusted selection to pick,
  because raw score equality is not the downstream semantic requirement.

Slot verification and end-to-end quality answer different questions. A per-call error can
fit a slot tolerance yet compound across layers, so qualification still uses candidate-
free pristine T evidence. Conversely, the candidate cannot redefine its local reference
by pointing at the current incumbent, which may itself contain prior proposals.

## Capability and fallback behavior

Before dispatch, the validator describes the live call and matches it against
the effective variant domain. Outside the declared domain, the trusted
incumbent path is used. That fallback is a safety property, but it cannot create
a win: a candidate that never runs, or runs only on immaterial calls, has no
positive marginal contribution.

Declare narrow domains honestly, then make sure diagnostic verification
actually exercises them. An exact model, phase, topology, dtype, or shape
predicate whose field is absent from the binding fails closed.

## Graph behavior

`graph_safe: true` is a routing declaration, not evidence. A crownable path must
have validator-produced graph observations for every applicable selected
variant and shape. CUDA host synchronization, data-dependent Python control
flow, allocations tied to replay values, pointer retention, or incomplete
replay writes will fail that stage. Continue with [Graph evidence](graph-safety.md).
