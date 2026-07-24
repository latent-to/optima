# Fidelity and quality authority

Optima uses different quality checks at different stages. Confusing them creates a major
security error: a useful development diagnostic is not necessarily safe as a grading
oracle.

## The quality layers

| Layer | Purpose | Crown authority? |
|---|---|---|
| Typed slot verification | Fast ABI, output-layout, numerical, applicability, and graph preflight | No |
| Contributor-controlled model A/B or in-engine audit | Development feedback and integration diagnosis | No |
| Registered audit-only role | Exact slot × TP-rank live-call evidence graded by the trusted host | Yes, when the arena policy requires it |
| Pristine T over sealed timed trajectories | Candidate-free, retained production quality evidence | Yes, as one required part of qualification |

## Slot verification

Each registered slot owns a typed input/output contract and correctness mode. Depending
on the slot, verification may use all-close tolerances, matched ratio, cosine similarity,
or top-k overlap. The verifier jitters eligible dimensions, checks variant routing, and
requires graph evidence where applicable.

This catches many broken kernels cheaply, but it samples a finite contract. It cannot by
itself establish end-to-end model behavior or serving quality.

## Development quality evidence

Engine developers may compare aligned rollout distributions, task behavior, and sampled
dispatcher calls while integrating a contribution. KL-like measurements are interpretable
only when stock-vs-stock variation is within the setup's calibration; launch
nondeterminism can otherwise dominate the signal. Tail and argmax-disagreement measures
help expose sparse corruption that a mean alone can hide.

An in-engine audit samples live candidate calls and compares them with a stock
calculation. It is valuable for debugging nondeterministic stacks, but the candidate
process contains the audit machinery and cannot grade a hostile engine. Framework-mode
token matching has the same limitation. These checks are engineering tools, not a
registered target or crown authority.

## Registered audit-only role

An arena may require a sealed audit plan after the resident speed stage. This is not the
candidate-side mechanism above. The audit role has its own plan and runtime identity,
executes outside the charged B/C/B′/(C′/B″) reads, and emits an exact slot × TP-rank/PID
witness. Trusted-host regrading checks expected rank coverage, unique processes, minimum
call counts, and retained violations or protocol errors without importing PyTorch.

The durable witness canonicalizes live floating-point facts into stable decimal strings
before computing receipt identity. A missing rank, duplicate/reused process, insufficient
coverage, tampered decimal fact, or unreopenable audit artifact cannot be waived by a
candidate-side report.

### Worker controls and comparator margin

The sealed `SlotAuditPolicy` owns the sampling rate, validator seed, expected slots,
expected TP member count, and minimum calls. For the separate audit-only candidate
session, the isolated engine worker maps that authority into two process-local controls:

- `OPTIMA_SLOT_AUDIT` is the policy's integer parts-per-million sampling rate converted
  to a fraction in `[0, 1]`.
- `OPTIMA_SLOT_AUDIT_SEED` is the policy's validator seed converted to an integer. Every
  rank receives the same seed so a sampled collective baseline is entered by all ranks
  rather than deadlocking on rank-divergent sampling.

The worker leaves both values empty when no audit policy is present. Charged
B/C/B′/(C′/B″) sessions therefore have no audit sampling or audit receipt and retain
their sealed graph configuration. The audit-only session is eager and untimed; the
worker disables CUDA graphs for that role. An unexpected audit receipt in a charged
candidate session is a protocol error.

The tensor comparison still occurs inside the candidate engine through
[`optima/audit.py`](https://github.com/latent-to/cacheon/blob/main/optima/audit.py).
For `matched_ratio` slots it grades against
`max(0, SlotSpec.correctness.min_ratio - 0.005)`. This `0.005` audit margin is not a
new slot tolerance and must not be applied by `verify`: component verification compares
the candidate with a high-precision reference, while the live audit compares candidate
and stock low-precision results, so both audit operands carry rounding. The margin was
selected with honest/reference and residual-drop controls; changing it requires fresh
control evidence and review.

The environment variables and rolling receipt are only worker instrumentation. Running
that instrumentation during development does not create crown authority. Authority
requires the independently sealed audit-only plan, a distinct session from every timed
role, bounded transport out of the worker, and Torch-free host regrading through
[`optima/audit_gate.py`](https://github.com/latent-to/cacheon/blob/main/optima/audit_gate.py).
The host requires the exact slot × TP-rank/PID matrix, the per-member minimum, zero
violations, and zero comparison errors.

### Production audit canary

Before audit evidence can satisfy a production activation gate, run the complete current
qualification path on the exact production image, model, topology, target set, audit
policy, and runtime identities. Retain the following controls:

1. Run an honest candidate through primary qualification and the required lane-swapped
   reproduction. Every registered slot on every TP rank must meet the sealed minimum
   call count with zero violations and comparison errors, and both attempts must reopen
   independently.
2. Run the registered residual-drop sabotage candidate through the same audit path. Its
   typed audit witness must make the aggregate qualification a nonretryable failure.
3. Inspect every charged B/C/B′/(C′/B″) session. Both audit environment values must be
   empty, no audit receipt may appear, and the charged result must retain the graph mode
   sealed by its speed plan.
4. In copies of the retained artifacts, remove or alter one slot/rank receipt and alter
   one audit-policy or request binding. Reopen and downstream settlement validation must
   reject every mutation rather than accepting an auditless or mismatched report.

The retained canary receipt is evidence only for those exact identities. Passing these
controls does not close in-process tampering, audit-role fingerprinting, timed-workload
fingerprinting, or other accepted residual risks; activation binds the canary separately
from its residual-risk acceptance.

## Pristine T

Production qualification launches a separate candidate-free reference session after the
resident speed executors are quiescent and the required audit stage completes. T:

- reopens an empty evaluation stack with no proposal contributions;
- binds the same runtime/model/reference identity and frozen calibration;
- receives sealed timed-read prompt and trajectory identities from the trusted controller;
- teacher-forces those trajectories;
- emits bounded token-level teacher evidence; and
- runs the registered hidden quality work.

The controller then regrades raw evidence under the frozen metric policy. Candidate C
does not choose prompts, support tokens, thresholds, or the hidden judge. Incumbent B′ is
also untrusted and is never substituted for T.

## Calibration

A crown-authoritative `CalibrationManifest` is content-addressed and frozen. It binds:

- pristine reference manifest;
- arena, runtime, base engine, model, logical hardware, and workload;
- verification policy and controller distribution;
- raw calibration evidence and seeds;
- speed margin, noise multiplier, and maximum noise; and
- metric envelopes, candidate deltas, and any absolute floors.

Controls include expected stock/positive passes and negative failure. A provisional,
stale, incomplete, or context-mismatched calibration is not usable for a crown.

## Quality decisions

The registered policy can grade metrics such as argmax disagreement, support coverage,
teacher NLL, KL-derived measures, tail rate, and task score. Exact metrics depend on the
frozen arena calibration; documentation must not present one universal threshold as the
Optima quality contract.

Missing teacher coverage, wrong prompt/trajectory identity, tampered evidence, or
unreopenable calibration yields `NO_DECISION`. A measured candidate regression yields
`FAIL`. Quality `PASS` is still only one prerequisite alongside execution, graph, speed,
identity, and reproduction evidence.

## Honest limits

- Finite prompts and hidden tasks cannot rule out all shape or workload overfitting.
- A pristine implementation can contain bugs; it is an independent authority, not a
  mathematical proof.
- Calibration is specific to the registered runtime, model, hardware, and workload and
  must be redone when those identities change.
- Contributor-controlled KL, task, or audit output is useful evidence for engineers but
  has no settlement effect.
- Passing fidelity does not establish license, security review, or release readiness.

See [Evidence and replay](../security/evidence.md) for how raw quality products are
retained and reopened.

## Source anchors

- [Typed slot contracts](https://github.com/latent-to/cacheon/blob/main/optima/slots.py)
- [Tensor output specifications](https://github.com/latent-to/cacheon/blob/main/optima/tensor_spec.py)
- [Qualification quality model](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification.py)
- [Torch-free audit gate](https://github.com/latent-to/cacheon/blob/main/optima/audit_gate.py)
- [Pristine wire protocol](https://github.com/latent-to/cacheon/blob/main/optima/eval/reference_protocol.py)
- [Calibration authority](https://github.com/latent-to/cacheon/blob/main/optima/eval/calibration.py)
