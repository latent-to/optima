# Slots and contribution targets

The validator publishes two related registries:

- the **slot registry** defines callable seams and their ABIs;
- the **target catalog** defines crownable contribution identities.

Do not treat those words as synonyms. A slot is an execution boundary. A target
is a validator-owned reward unit over one or more slots.

You can print the slot registry from the checkout you are developing against:

```bash
python -m optima.cli slots
```

The authoritative sources are
[slots.py](https://github.com/latent-to/cacheon/blob/main/optima/slots.py) and
[target_catalog.py](https://github.com/latent-to/cacheon/blob/main/optima/target_catalog.py).

## Current slot catalog

There are 11 semantic slots. `entry` below means the callable named by your
manifest; it does not require the Python function itself to be named `entry`.

| Slot | Kind | Required call boundary | What the validator retains |
|---|---|---|---|
| `activation.silu_and_mul` | op | `entry(x, out)` | MLP activation output |
| `norm.rmsnorm` | op | `entry(x, weight, out, eps)` | pure RMSNorm output |
| `attention.sdpa` | block | `entry(q, k, v, out, sm_scale, causal)` | dense/GQA/MQA attention output |
| `attention.decode` | block | `entry(q, k, v, seq_lens, sm_scale, out)` | paged decode-attention output |
| `attention.msa_block_score` | block | `entry(q, index_k, seq_lens, block_size, out)` | decode-time block scores; validator owns top-k selection and attend |
| `attention.msa_prefill_block_score` | block | `entry(q, index_k, prefix_len, scale, block_size, out)` | causal per-row prefill block scores; validator owns top-k selection and attend |
| `moe.fused_experts` | block | `prepare(w13, w2)` plus `entry(x, topk_ids, topk_weights, prepared, out)` | local expert result; stock path owns the trailing reduction |
| `moe.fused_experts_reduce` | collective | `prepare(w13, w2)` plus `entry(x, topk_ids, topk_weights, prepared, out, group)` | already reduced expert result |
| `collective.all_reduce` | collective | `entry(x, out, group)` | sum across the supplied process group |
| `collective.ar_residual_rmsnorm` | collective | `entry(x, residual, weight, eps, out_norm, out_residual, group)` | reduced residual and normalized output |
| `collective.moe_finalize_ar_rmsnorm` | collective | `entry(gemm_out, row_map, scales, residual, weight, eps, out_norm, out_residual, group)` | finalized/reduced residual and normalized output |

The two MSA slots output **scores**, not selected indices or attended values.
The production engine's runtime top-k controls the validator-owned selection tail.
`SlotSpec.correctness.top_k` is a separate verification parameter: it selects the
width of the score-sheet overlap comparison (the registered width is eight blocks), but does not
configure production selection or gate whether the score kernel routes. A runtime
call may therefore use a different top-k while the candidate still fills the complete
score sheet and the stock tail applies the engine's value.

The prefill slot is a distinct long-prefill boundary and its canonical live
descriptor is eager; do not assume a decode optimization exercises it.

Registration and live installation are also different facts. The
`attention.msa_block_score` decode contract is registered and CPU-verifiable, but its
current SGLang decode adapter is deliberately a non-installing stub until the pinned
runtime has a stable model-specific chokepoint. The prefill sibling has a real guarded
adapter. Confirm that the operator's arena actually binds and activates the target before
investing in a production submission; a contract alone does not make a call site hot.

Collective slots are distributed contracts. `group` is the process group the
validator supplies, and every listed output is validator-allocated. Test with
the arena's world/TP size, not just one rank.

See [Kernel ABI](kernel-abi.md) for tensor semantics and
[Graph evidence](graph-safety.md) for capture requirements.

## Singleton targets

The current default target catalog registers one singleton target for each of
the 11 slots. Its target ID is the slot ID. A normal proposal therefore names
the slot target explicitly:

```toml
[competition]
target = "attention.msa_prefill_block_score"
mode = "slot"
```

The catalog, not your manifest, binds that target to its member slot, ABI,
reference, verification profile, serving binding, correctness policy, and
allowed implementation features.

## The registered atomic target

The default catalog also registers:

| Target | Mode | Members | Displaces |
|---|---|---|---|
| `collective.moe_epilogue.v1` | `atomic` | `collective.ar_residual_rmsnorm`, `collective.moe_finalize_ar_rmsnorm` | both corresponding singleton targets |

Use an atomic target only when the optimization's semantics genuinely require
the coupled boundary and your bundle implements all registered members:

```toml
[competition]
target = "collective.moe_epilogue.v1"
mode = "atomic"
```

Adding two unrelated `[[ops]]` rows does not create a target. Nor may a miner
declare membership, displacement, overlap, or a new target ID. An unregistered
combination belongs in the [Discovery lane](discovery-lane.md).

## Selecting a target

Start from the published arena, not from an isolated kernel idea:

1. Confirm that the arena activates the target, model, architecture, dtype,
   topology, and serving phase you intend to optimize.
2. Profile the exact incumbent stack and find a material wall-time boundary.
3. Choose the smallest registered target that contains the required delta.
4. Describe every specialization in an explicit capability domain.
5. If the change needs engine-wide setup, arbitrary SGLang edits, or semantics
   outside a registered target, stop and use discovery instead.

For a first implementation, `activation.silu_and_mul` and `norm.rmsnorm` have
the smallest single-process ABIs. They are good learning targets, not a promise
of economic headroom against tuned incumbents. Advanced collective and deep-MoE
targets require the matching multi-GPU and build environment to test honestly.

### A decision procedure

Walk the desired change from semantics outward:

1. **Name the changed outputs.** Which validator-visible tensor values differ from the
   incumbent implementation while remaining semantically equivalent?
2. **Find who owns those outputs.** Locate the narrowest slot whose ABI owns every changed
   value and no unrelated engine behavior.
3. **Check the live arena binding.** A registered target must also be active for the exact
   model, runtime, architecture, dtype, phase, topology, and graph regime you plan to
   optimize.
4. **Check required features.** If the implementation needs an override point, CUDA
   rebuild, dependency patch, or multiple members, the selected target must explicitly
   allow those observed features.
5. **Estimate materiality.** Profile the incumbent and calculate whether improving that
   boundary could move end-to-end critical-path time above calibrated noise.
6. **State the honest domain.** Use capability predicates for real specialization
   boundaries. Do not use them to hide failing shapes or claim a target that never routes.

The outcome should be one of exactly three shapes: one registered singleton, the exact
complete member set of a registered atomic target, or discovery. “Closest available
slot” is not a fourth option.

### Worked choices

| Idea | Choice | Reasoning |
|---|---|---|
| fuse SiLU and multiply for a particular token range | `activation.silu_and_mul` singleton with a constrained variant | both operations and the output are already inside one slot |
| add a residual connection to pure RMSNorm | not `norm.rmsnorm`; use a matching registered collective boundary only if its full semantics apply, otherwise discovery | the singleton RMSNorm contract explicitly does not own residual addition |
| replace local expert compute and its trailing reduction as one implementation | `moe.fused_experts_reduce` | this slot, unlike `moe.fused_experts`, owns the supplied-group reduction |
| jointly alter the shallow and deep MoE collective epilogues | `collective.moe_epilogue.v1`, implementing both members | the catalog already registers and prices the coupled overlap unit |
| patch scheduler batching or invent a new attention seam | discovery | engine control flow lies outside every component callable ABI |

This exercise prevents two common errors. Choosing a boundary that is too narrow makes
the desired optimization impossible without hidden side effects. Choosing one that is
too broad destroys the causal comparison: the validator can no longer attribute the
measured change to one registered reward unit.

## Target resolution fails closed

Production intake resolves the selected delta against trusted observations of
its features. A claim can be rejected when, for example:

- the target is unregistered or `mode` disagrees with the catalog;
- the bundle's implemented member set does not equal the target;
- an atomic claim omits a member;
- the bundle uses a feature the target does not allow;
- observed rebuild features are incomplete;
- multiple variants have overlapping capability domains;
- an `ops.setup` hook appears in a registered target (none currently allow it).

This is why a manifest that merely parses is not necessarily a crownable
proposal. Continue with [Bundle format](bundle-format.md).
