# Graph evidence

CUDA-graph behavior is part of the serving contract, not a performance toggle
the candidate may opt out of. A kernel that works eagerly but fails capture or
replay cannot replace an incumbent graph path.

## Capture and replay in plain language

Normal eager execution lets Python and the runtime decide what to launch on every call.
A CUDA graph records one launch sequence against fixed storage addresses, then replays
that sequence with new tensor contents. Serving uses this to remove repeated CPU launch
overhead and stabilize scheduling.

For a candidate, the important phases are:

1. **Warmup:** imports, legal lazy initialization, and approved compilation complete
   before evidence-bearing capture.
2. **Eager check:** the callable produces the correct result outside a graph. A failure
   here is ordinary ABI/math failure, not a graph problem.
3. **Capture:** operations issued by the callable must be legal on the capture stream.
   Host synchronization or data-dependent Python decisions break this phase.
4. **Replay:** the validator changes logical inputs, poisons outputs, replays without
   rerunning candidate Python, and compares the new result. A kernel which captured a
   stale pointer or first-call value fails here.

Graph safety is not synonymous with “uses CUDA.” It means the implementation continues
to compute the slot contract when Python launch logic is frozen and only device work is
replayed.

## Declaration is not proof

This metadata:

```json
{"graph_safe": true}
```

only declares that the variant may be routed in `cuda_graph` mode. It does not
prove capture safety, satisfy qualification, or override a failed observation.

The validator creates a graph requirement bound to all of the following:

- target specification and every target member;
- selected candidate delta and candidate launch identity;
- slot and variant identity;
- the exact shape-descriptor set and applicability projection;
- the required replay count.

It then stores raw, content-addressed observations and regrades them. Qualification
does not trust one aggregate `graph_passed` boolean supplied by a worker.
The schema and veto logic are in
[qualification.py](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification.py),
and finalized-intake projection is in
[qualification_intake.py](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification_intake.py).

## What each observation proves

For every required variant and shape, graph evidence records:

- whether the variant and shape were applicable;
- whether eager execution passed;
- whether a graph was required;
- how many replays completed;
- whether replay outputs passed;
- a failure class such as eager execution, capture, or replay.

Coverage is as important as a successful sample. Missing members, variants, or
shapes produce `NO_DECISION`; applicability disagreement, incomplete domain
coverage, an applicable failed shape, or a member with no applicable passing
shape fails the veto. A replay-count mismatch is also not a pass.

In short: every selected implementation path must be evidenced. A fallback that
silently makes the candidate N/A cannot create a crown.

## Local graph diagnostics

On CUDA, `verify` graph-tests op slots by default. For block and collective
variants, graph testing follows the effective `graph_safe` declaration:

```bash
python -m optima.cli verify my_bundle \
  --device cuda --dtype bfloat16
```

For a collective:

```bash
python -m optima.cli verify my_collective \
  --device cuda --dtype bfloat16 \
  --world-size 4 --tp-size 4
```

The collective verifier creates the actual rank group, captures each applicable
clean-room shape, poisons outputs between replays, and grades every rank. Run it
on homogeneous GPUs matching the arena architecture.

A CPU `verify` can prove eager numerical behavior, but it cannot produce CUDA
capture evidence. Likewise, a local CUDA pass is a developer diagnostic—not
the authority-bound evidence retained by production qualification.

When reading local output:

- `graph=not-required` says this run did not request graph proof; it does not certify an
  eager-only implementation for an arena that requires graphs;
- `graph=verified` plus positive `graph_replays` says the local profiles passed capture
  and checked replay;
- `NUMERICAL_PASS ... graph=NOT_VERIFIED` means the math passed but the requested graph
  contract did not; and
- N/A profiles add no graph evidence because the candidate was not applicable.

Production needs the authority-bound shape and applicability projection, so even a local
`graph=verified` is preparation rather than a qualification receipt.

## Common capture failures

Graph-safe serving code must avoid:

- `.item()`, device-to-host reads, or synchronizing `.cpu()` calls;
- Python branches whose path depends on live tensor values;
- dynamic allocation or compilation during capture/replay;
- retaining a pointer to one output and writing it on later calls;
- reading stale tensor values captured from a previous invocation;
- writing only part of a poisoned output;
- changing collective order across ranks;
- creating or destroying process groups in the entry callable;
- using a different stream or event protocol without capture-safe ownership.

Prepare layouts, compile approved artifacts, and allocate persistent workspace
before serving capture. During replay, consume the current dynamic inputs and
write the current validator-provided outputs.

One subtle failure pattern is a kernel that appears stable because the shape stays fixed
while values change. CUDA graphs intentionally support that regime: storage addresses
remain fixed, but sequence lengths, routing IDs, scores, and tensor data can differ on
each replay. Treat any such value as live device input. Reading it into Python during
capture freezes one branch and is semantically wrong even if the first replay happens to
match.

## Variable shapes and variants

CUDA graphs fix storage addresses, not semantic values. The verifier replays
fresh logical input values through the captured path. A kernel that bakes in
first-replay data can appear correct at capture and fail replay.

If algorithms genuinely differ by shape, use explicit disjoint variants and
capability domains. Do not branch on a host read of a runtime tensor. The graph
requirement covers every selected variant and all of its applicable descriptor
profiles.

The MSA prefill call descriptor includes `graph_mode = "eager"` for its current
live binding, but that is not permission to invent an exemption. Crownability is
decided by the validator's published target/graph requirement, and the current
graph veto requires complete positive evidence for selected members. Confirm the
arena requirement before investing in an eager-only specialization.

## Do not benchmark a different regime

Disabling CUDA graphs in a contributor-controlled launch can help debug startup or math,
but it changes the incumbent execution regime and therefore cannot establish a production
speedup. A graph-on candidate must be compared with the graph-on incumbent under the same
evaluation stack.

When graph verification fails, use the stage-specific guidance in
[Diagnostics](diagnostics.md). Do not work around it by narrowing metadata until
the candidate never executes.
