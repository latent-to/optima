# MiniMax-M3 engineering results

The MiniMax-M3 campaigns established that bounded collective epilogues can
produce meaningful whole-engine serving improvements. They also exposed the
measurement and authority gaps that motivated the current resident referee.

These are dated engineering results. They are not a current-path crown,
production release, or activation receipt.

## Workload and target

The campaign used MiniMax-M3-NVFP4 at tensor parallelism four on B300-class
hardware. The strongest proposals fused the MoE tail where expert output,
collective reduction, residual addition, and RMS normalization otherwise cross
multiple launch and communication boundaries.

That work informed three registered identities:

- `collective.ar_residual_rmsnorm`;
- `collective.moe_finalize_ar_rmsnorm`; and
- atomic `collective.moe_epilogue.v1`, which owns and displaces both singleton
  identities when they form one inseparable change.

The atomic target prevents one fused implementation from earning duplicate
economic titles through overlapping views of the same delta.

## Historical measurements

Under the pre-resident qualification harness, the shallow fused path measured
`1.044×` and `1.049×` in two runs. The deeper fused path measured `1.074×` and
`1.071×`. A later chain-loop reproduction measured `1.072×` for the deep bundle.
The retained campaign notes report zero sampled audit violations.

These measurements support the engineering hypothesis that the optimization
surface is real. They do not satisfy the current authority model because they
predate:

- a reviewed injected arena service;
- persistent routing-only resident screening;
- adaptive two-lane resident crossover qualification;
- exact physical-lane role swap for independent reproduction;
- torch-free host audit regrading over exact slot × TP-rank coverage;
- current-schema evidence reopen and transactional settlement; and
- the separate integrated release and serving stack.

## Lessons incorporated into the implementation

### Optimize the complete owned boundary

A reduce-only slot cannot express useful overlap with its producer. Giving
`moe.fused_experts_reduce` ownership of the trailing reduction and defining a
bounded fused epilogue target creates a semantic unit large enough for
communication and launch optimization while preserving a validator-owned output
contract.

### Direct rollout comparison is not universal quality authority

Large low-bit MoE launches exhibited noisy logit differences between nominally
identical engines. The current design therefore seals controller-observed speed
trajectories and grades quality with a separate candidate-free pristine reference.
MiniMax-M3 additionally requires its registered audit authority; rollout KL remains
advisory for this arena.

### Graph behavior is part of correctness

An eager-correct kernel can disappear under graph capture or change behavior on
replay. Current verification binds applicable entries and shapes, refreshes dynamic
inputs, poisons outputs, and checks replay while authoritative qualification observes
the graph-enabled complete engine.

### Candidate code cannot own the referee

Native extensions and Python hooks must remain outside trusted timing and grading.
Disposable build workers and isolated OCI execution reduce that boundary, while the
host owns arm assignment, clocks, protocol evidence, teardown, audit grading, and
the candidate-free reference.

### Reproduction is protocol authority

Two numbers in a worklog are not two independent qualification attempts. Current
settlement requires distinct authority and evidence for the second PASS and, for
resident qualification, an exact swap of the baseline and candidate physical TP
lanes. Settlement uses the lower reproduced speedup.

## Current interpretation

Read the historical B300 numbers as discovery and design evidence. The maintained
implementation now has CPU/mock coverage for the resident adaptive path, audit
receipt regrading, settlement, and emissions state machines, but it does not retain
a current-path production MiniMax-M3 crown.

The remaining empirical gates include:

- an exact production B300 canary for the current resident primary and lane-swapped
  reproduction path;
- accepted residual-risk authority for in-process tampering, audit-role
  fingerprinting, and timed-workload fingerprinting;
- a production engine image served end to end through the reviewed release path;
- exact campaign, family, membership, and reserve manifests; and
- a live V2 activation and confirmed debt-publication receipt.

See [Current status](../reference/state-of-record.md) for the maintained evidence
ledger, [Authoritative qualification](../validator-guide/qualification.md) for the
current protocol, and [Emissions policy](../reference/emissions-policy.md) for the
economic contract.
