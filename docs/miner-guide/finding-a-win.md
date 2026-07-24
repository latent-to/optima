# Finding a marginal win

The useful question is not “can this kernel beat a reference implementation?”
It is “does this exact registered delta improve the current incumbent serving
stack on the published arena workload, while preserving required behavior?”

Most faithful first attempts are slower. Tuned inference stacks have already
captured obvious local optimizations, so credible wins usually come from a
measured boundary mismatch: fusion, communication overlap, layout conversion,
specialization, or redundant traffic the incumbent cannot avoid.

## Start with the arena contract

Before profiling, record the validator-owned context:

- evaluation stack digest and incumbent generation;
- model/reference identity and SGLang substrate;
- active target catalog and target contract;
- architecture, TP/EP topology, quantization, dtype, graph mode;
- request mix, input/decode lengths, concurrency, batching, and cache policy;
- quality and speed calibration identities.

If these facts are not published, you cannot know whether a local result targets
the active reward family. Never optimize a convenient local stack and silently
generalize its result to another arena.

## Profile wall time, not kernel names

Use a systems trace of the complete incumbent engine under the target workload.
Group time by semantic boundaries and critical path, not by a sum of overlapping
CUDA durations. Concurrent compute and communication make summed kernel time a
misleading denominator.

For a slot with critical-path fraction `f`, made `k` times faster in isolation,
the idealized upper bound is:

```text
end_to_end_speedup <= 1 / ((1 - f) + f / k)
```

That is still optimistic: dispatch, memory traffic, synchronization, graph
shape, batching, and downstream effects can erase the gain. If the bound is
below the arena's resolvable noise, choose another target.

### A worked materiality estimate

Assume a trace shows that a slot accounts for 8% of critical-path wall time and your
microbenchmark makes its callable twice as fast. The optimistic bound is:

```text
1 / ((1 - 0.08) + 0.08 / 2) = 1.0417
```

So the largest plausible end-to-end improvement is about 4.17%, before integration
overhead or secondary bottlenecks. That is worth an end-to-end experiment. If the same
kernel occupied 0.5% of the critical path, even deleting all of its time would cap the
gain near 0.5%; on a noisy arena that may not be a measurable reward opportunity.

Now check causal ownership. If half of the observed 8% is an adjacent layout conversion
outside the slot, your bundle cannot claim that half unless a registered target owns it.
Use the smaller in-boundary fraction in the estimate. The calculation is a filter for
where to invest, not evidence the candidate won.

## Look for boundary-shaped opportunities

Promising classes include:

- fuse operations already contained in one registered slot;
- overlap expert compute with the trailing reduction in
  `moe.fused_experts_reduce`;
- reduce memory round-trips inside a collective epilogue;
- specialize a variant for a provably disjoint shape/topology domain;
- improve a model-specific score or quantized epilogue while retaining the
  validator-owned selection/downstream path;
- replace repeated layout work with target-approved load-time preparation;
- use the registered atomic target when one coupled deep delta genuinely owns
  both of its member seams.

Do not expand the manifest until a desired optimization “fits.” The target
catalog fixes the smallest allowed delta. If the change needs an unregistered
engine seam, scheduler behavior, broad source patch, or engine-wide setup, move
it to [Discovery](discovery-lane.md).

## Match the workload regime

Decode, short prefill, and long/chunked prefill are different performance
problems. So are TP topologies with and without fast peer links.

For attention work:

- `attention.msa_block_score` is a decode-side block-score boundary;
- `attention.msa_prefill_block_score` is the causal per-row long-prefill
  boundary;
- an optimization to one should not be measured on a workload dominated by the
  other.

A developer diagnostic can generate longer inputs, but repeated timed prompts
with radix caching enabled may turn later passes into cache hits. For a prefill
diagnostic, reproduce the arena's cache policy or explicitly disable the radix
cache in both arms. Treat this as local debugging, not authority.

For collective work, reproduce the actual world/TP size and interconnect. A
single-GPU benchmark says nothing about distributed ordering, graph replay, or
critical-path overlap.

## Compare against the incumbent, not “stock”

An active stack can already contain crowned contributions on other targets. The
marginal baseline is that exact incumbent stack, with its exact engine release,
build products, model assets, and launch controls.

The authoritative bracket is:

```text
incumbent = exact incumbent engine, loaded once on the resident baseline lane
candidate = same stack with exactly the selected target delta, loaded once on
            the disjoint resident candidate lane
B → C → B′ [→ C′ → B″] = serialized timed reads; repeats are conditional
A = registered eager, untimed candidate audit
T = candidate-free pristine quality reference after candidate teardown
```

Changing a backend, graph flag, prompt mix, topology, or engine option only in C
does not measure a component delta. Ad hoc developer launches may expose such
controls for diagnosis, but authoritative qualification binds the complete v3
resident speed, audit A, and pristine T plan from the validator's stack,
qualification, and reference manifests.

The repository implementations are
[`stack_plan.py`](https://github.com/latent-to/cacheon/blob/main/optima/stack_plan.py)
for incumbent/candidate stack construction and
[`scoring.py`](https://github.com/latent-to/cacheon/blob/main/optima/eval/scoring.py)
for conditioned serving-score calculation.

## Build the narrowest honest capability domain

A specialized variant should state precisely where it wins and remains correct:
model, architecture, dtype, quantization, phase, topology, and relevant shape
ranges. Outside that domain the incumbent safely serves the call.

Narrowing the domain after seeing failures is legitimate only when it describes
a real algorithmic boundary and still covers material arena calls. A variant
that never executes has no marginal benefit; overlapping variants are rejected.

Write a small target worksheet before implementation:

| Question | Example answer |
|---|---|
| target and active binding | `activation.silu_and_mul` on the operator's published arena generation |
| critical-path fraction | 8% in graph-on decode at the published concurrency |
| proposed mechanism | fuse activation/multiply and reduce one global-memory round trip |
| exact domain | BF16, `sm90`, decode, token counts 1–256 |
| fallback | incumbent row for every non-matching descriptor |
| local correctness proof needed | all applicable slot shapes, mutation/output checks, CUDA replays |
| end-to-end falsifier | no throughput change, B/B′ drift, fallback on material calls, or quality regression |

This worksheet requires a causal chain from the measured bottleneck to a selectable
variant. “My kernel is 30% faster” is incomplete until the replaced fraction of the arena
path and the domain's routing frequency are specified.

## Use a disciplined development ladder

1. `scan`: catch structural and static-policy errors.
2. `verify`: test ABI, numerical behavior, mutation, applicable shapes, and CUDA
   graph replay where available.
3. Microbenchmark the exact callable to understand the mechanism, not to claim
   end-to-end speedup.
4. Trace the complete incumbent and candidate developer launches.
5. Run a matched graph-on whole-engine diagnostic using the
   [canonical local profiling record](../validator-guide/running-evals.md#performance-development)
   and inspect B/C/B′ drift.
6. Run the relevant non-authoritative quality diagnostics.
7. Repeat across fresh processes, seeds, and conditioning until the effect is
   larger than normal variance.

Steps 1–7 are miner-side diagnostics. Only finalized, isolated, identity-bound,
independently reproduced validator qualification can crown the proposal.

## Interpret local outcomes correctly

- A faster microkernel with flat end-to-end throughput is not a win; the slot
  was not critical or the gain moved elsewhere.
- A faster C with disagreeing B/B′ is measurement uncertainty, not a pass.
- A speed gain with failed quality is a failed candidate, not a tradeoff score.
- A graph-off gain is evidence about debugging mode, not the arena.
- A pass on one topology is not evidence for an undeclared topology.
- A first production PASS becomes `reproduction_pending`, not crowned.

Also distinguish failure from uncertainty. A consistently slower C or a quality
regression is a candidate `FAIL`; revise the source and create a new identity. Disagreeing
B/B′ bookends, missing evidence, or infrastructure faults can be `NO_DECISION`; they do
not justify changing the math until the operator's receipt identifies a candidate fault.

The [MiniMax-M3 results](../results/minimax-m3.md) show how profiling identified
fused collective boundaries. Those measurements do not substitute for
qualification under the current target, stack, evidence, and two-pass
authority.

Once the mechanism survives this scrutiny, prepare the source-only archive and
follow [Submitting](submitting.md).
