# Measurement and decision policy

Production Optima does not reduce a proposal to one self-reported score. It derives a
three-way qualification decision from retained execution, graph, speed, and pristine
quality evidence, then requires independent reproduction before settlement.

## Screens are not scores

Static, build, ABI, graph, and abbreviated-serving screens protect scarce evaluator
capacity. They may promote, reject, retry, or hold a proposal, but are marked
non-crownable. Their timings and grades cannot update the evaluation stack.

## Marginal comparison

For a registered target, the v3 production policy constructs two isolated resident TP
lanes:

- B: the exact frozen incumbent stack;
- C: that same stack with one registered target replaced; and
- B′: a second incumbent read from the same resident baseline lane.

Clear results stop after resident B/C/B′. A policy-defined borderline result
adds C′ and B″ without reloading either engine. During execution, the controller
fixes prompt batches and token budgets, serializes the two lanes, validates
bounded batch frames and token numerators, and constructs charged rates that
include registered conditioning and timed intervals. The durable resident
witness retains the actual three-or-five-read schedule, physical-lane authority,
operational timing, and budget. After the speed lifetimes are quiescent,
qualification runs registered eager audit A when the plan requires it, destroys
candidate lifetimes, and then runs pristine T.
Reopen recomputes tokens/second and the frozen decision from typed counts and intervals,
not from raw session frames. Candidate-reported aggregate throughput and resident-screen
rates are never accepted as authority.

The speed estimate is conceptually:

```text
speedup = candidate_rate / mean(baseline_before_rate, baseline_after_rate)
noise   = relative spread of the two baseline rates
bar     = 1 + max(min_margin, noise_multiplier * noise)
```

The exact thresholds come from a frozen `CalibrationManifest` bound to the reference,
arena, runtime, model, hardware, workload, verifier, and controller distribution. If
baseline disagreement exceeds the calibrated maximum, the result is `NO_DECISION`.

## Complete qualification decision

A candidate can pass only when all required products agree:

| Product | Failure meaning |
|---|---|
| Execution evidence | Wrong/missing role, launch, device, protocol, or completion |
| Graph evidence | Missing target member/variant/shape coverage or capture/replay failure |
| Speed evidence | Below the calibrated bar, or too noisy to decide |
| Audit-only evidence | Missing slot × rank/PID coverage, retained violation, or protocol error |
| Pristine quality evidence | Regression against frozen metric envelopes or hidden work |
| Identity checks | Evidence does not describe the committed arena, stack, target, or delta |

Attributable violations yield `FAIL`. Infrastructure, missing evidence, stale identity,
or excessive drift yields `NO_DECISION`. Only complete green evidence yields `PASS`.

## Independent reproduction

The first `PASS` moves the reservation to `reproduction_pending`. A second `PASS` must
match the economic identity while using distinct authority, attempt, report, and
selection evidence. The settlement candidate's conservative speedup is:

```text
settled_speedup = min(primary_speedup, reproduction_speedup)
```

There is no single-pass fast path to a crown.

For resident v3, reproduction must also swap the baseline and candidate physical TP-lane
orientations exactly while retaining the same speed-policy and settlement-control
digests.

Here “independent” means the seven required authority, plan, attempt, report, commitment,
secret-commitment, and selection-evidence digests differ. It does not by itself prove
separate operators, hosts, or infrastructure failure domains.

## Settlement cohort over one incumbent authority

The store leases one economically unblocked group sharing a qualification authority and
one exact incumbent stack. Stale candidates are held. Across all current registered rows
in that leased group—even rows for non-overlapping targets—the planner selects one winner
by conservative speedup and uses finalized arrival order as the tie-break. The shared
incumbent advances once, so every other current row is held for a fresh qualification
against the new stack rather than treated as an independent per-target argmax.

The winning transaction may emit crown, retirement, neutralization, adoption, and stack
transition events. Atomic targets explicitly displace overlapping singleton targets;
manifest order and bundle packaging never decide overlap.

Discovery is different: a qualifying discovery may create a bounded bounty event, but it
does not install a stack manifest or create a standing reward family.

## Reward policy follows the activated generation

Under retained legacy V1 authority, each active registered target defines one
reward family. The policy derives standing credit from reproduced marginal
improvement and age. The normative conversion, decay equation, and integer
rules live in
[Legacy V1](../reference/emissions-policy.md#legacy-v1).

Standing-claim age begins at the proposal's finalized submission block, which
settlement stores as `crowned_block`; discovery lifetime likewise begins at
that submission block via `awarded_block`. Qualification or settlement delay
never resets reward age.

An active atomic target suppresses overlapping singleton families. Packaging,
integration, and release records do not create additional families. Discovery bounties
are non-renewable, expire, and share a policy-bounded pool.

The final multi-arena projection is exact integer ppm and is built only after every
active family reopens against current stack and metagraph authority. A stale or missing
active claim holds the entire projection rather than silently redistributing its share.

Finite-debt V2 does not reuse this standing-decay formula. It issues finite principal
from reproduced log-relative improvement and a family clock, composes registered-CROWN
and reviewed-discovery debt, and debits only after confirmed chain readback. The two
generations remain separately versioned and never publish concurrently. See
[Emissions policy](../reference/emissions-policy.md).

Read [Settlement and weights](settlement-and-weights.md) for transaction and publication
details.

## What a result means

A crown means: under the registered arena, workload, calibration, and two attempts meeting
the seven digest-distinctness checks, the exact delta improved the exact incumbent with
acceptable measured quality.

It does not mean:

- the contribution improves every model, topology, or traffic mix;
- the measured speedup is a service-level capacity guarantee;
- the proposal is licensed, maintainable, reviewed, or ready to ship; or
- any score produced outside registered, retained qualification has economic effect.

## Source anchors

- [Raw speed recomputation](https://github.com/latent-to/cacheon/blob/main/optima/eval/scoring.py)
- [Frozen calibration](https://github.com/latent-to/cacheon/blob/main/optima/eval/calibration.py)
- [Qualification runner](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification_runner.py)
- [Resident crossover](https://github.com/latent-to/cacheon/blob/main/optima/eval/crossover_runtime.py)
- [Settlement](https://github.com/latent-to/cacheon/blob/main/optima/settlement.py)
- [Economics](https://github.com/latent-to/cacheon/blob/main/optima/economics.py)
