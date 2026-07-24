# Copy and priority controls

Optima separates three questions:

1. Which proposal arrived first under chain consensus?
2. Are two submitted deltas authoritatively the same for automatic disposition?
3. Do two proposals look similar enough to merit human review?

These questions must not be collapsed into a fuzzy plagiarism score.

The policy is intentionally asymmetric in time but narrow in content: finalized order
decides which of two authoritative matches is earlier, while fingerprints compare only
the miner-supplied attributable delta. The validator-supplied incumbent must not make all
later proposals look like copies of whichever candidate was evaluated first.

## Finalized priority

The miner commits a content hash and HTTPS URL through the chain's native timelock
commit-reveal. While the timelock mechanism holds, other participants cannot read the
hosted proposal from the on-chain payload before reveal.

The validator reads finalized reveal history and records canonical block, event, and
subevent positions before fetching anything. Copy priority therefore comes from chain
order, not HTTP latency, evaluator scheduling, or the validator's wall clock.

The complete finalized history is reserved; the current loop does not keep only each
hotkey's latest reveal.

## Fingerprint scope

`SubmittedDeltaFingerprint` covers miner-supplied attributable bytes only:

- product kind (`component` or `discovery`);
- registered target and target-spec identity for a component;
- exact target members;
- exact selected payload and delta digests;
- normalized delta identity;
- substantial per-file containment fingerprints; and
- advisory structural fingerprints.

The incumbent stack, validator adapters, base runtime, and canonical materialized engine
bytes are excluded. Two miners must not match merely because the validator assembled both
against the same incumbent.

## Automatic copy decisions

For an earlier and later delta in the same reward namespace, these comparisons are
authoritative:

| Match | Automatic result |
|---|---|
| Exact selected payload identity | Later delta is a copy |
| Equal normalized whole-delta identity | Later delta is a copy |
| Symmetric containment of substantial file fingerprints | Later delta is a copy |

Normalization removes cosmetic Python formatting/comments/docstrings, normalizes
declared CUDA text, follows bundle-local Python imports, includes all explicit variants,
and includes declared dependency patches. Symmetric containment prevents an extra padding
operation or fresh sibling variant from hiding a stolen target implementation.

The intake store compares only earlier finalized proposals from another hotkey. Copy
reconciliation is durable and idempotent. If an earlier fingerprint becomes available
after a restart window, a later unresolved or even previously qualified copy can be
corrected before economic settlement.

### Worked comparisons

Assume hotkey A finalizes before hotkey B in the same registered reward namespace:

| A and B relationship | Policy result | Reason |
|---|---|---|
| Byte-identical selected target payload | B is an automatic copy | Exact selected identity |
| Same implementation with comments, docstrings, and cosmetic formatting changed | B is an automatic copy when normalized whole-delta identity matches | Cosmetic changes do not create new attributable work |
| A's substantial implementation copied into B with one unused padding file | B is an automatic copy when symmetric containment matches | Padding cannot hide containment |
| Similar Python control-flow skeleton but different authoritative delta identities | Advisory review signal only | Structural similarity is intentionally non-dispositive |
| Both candidates include the same validator-supplied incumbent in their materialized engines | No match from that fact | Incumbent bytes are outside submitted-delta scope |
| Independently convergent deltas normalize identically | B receives the deterministic later-copy disposition | Automatic identity cannot infer intent; finalized priority resolves the tie |
| B is a discovery patch while A is a registered component | Compared within their separate namespaces, not collapsed by related purpose | Product kind and reward namespace are identity inputs |

Automatic disposition says nothing about whether A will pass qualification. It prevents B
from acquiring later economic priority for the same attributable bytes; it does not turn A
into a winner or into reviewed product source.

## Advisory similarity

Shared fragments and structural Python skeletons are **not** automatic demotion signals.
They can arise from common utility code, simple contracts, or independent implementation.
The comparison records them as advisory context for review.

Structural normalization blanks Python identifiers and constants while retaining
operations and attribute structure. The automatic signal has no equivalent semantic CUDA
parser. Renaming CUDA identifiers and changing constants can evade that advisory layer.

## Discovery

Discovery proposals use their separate closed manifest and declared patches. Copy
identity is derived from the exact proposal plus normalized patch content. Discovery
still occupies a separate reward namespace and cannot collide with a registered
component merely because both touch related upstream code.

## What copy detection does not do

- It does not prove legal authorship, license compliance, or copyright ownership.
- It does not detect every semantic rewrite, obfuscation, generated source, or CUDA
  near-copy.
- It cannot distinguish independent convergence when two deltas normalize identically;
  finalized priority is the deterministic policy in that case.
- It does not award a crown. Non-copy candidates still need screens, two qualifications,
  and transactional settlement.
- It does not create reward by splitting one implementation across identities; rewards
  attach to active target contributions, not raw submission count.

False-positive cost is why fuzzy structural similarity remains advisory. Expanding the
automatic rule requires a reviewed policy and regression tests, not an operator's ad hoc
similarity threshold.

## Reviewer and operator checklist

When investigating a suspected copy:

1. Reopen finalized block/event/subevent authority for both proposals before comparing
   arrival times or evaluator logs.
2. Confirm both fingerprints use the same product/reward namespace, exact target and
   target-spec context, and submitted-delta scope.
3. Separate exact, normalized, and symmetric-containment matches from advisory structural
   fragments in the report.
4. Reopen the immutable published bundles and recompute fingerprints; do not compare a
   mutable hosted URL after the fact.
5. Exclude validator-supplied incumbent, adapter, and base-runtime bytes from the claimed
   overlap.
6. Persist the disposition through the intake state machine and verify it before
   settlement. Do not hand-edit a crown or weight row.
7. Route authorship, license, or semantic-plagiarism disputes to human/legal review; the
   copy policy does not establish those facts.

## Source anchors

- [Submitted-delta fingerprints](https://github.com/latent-to/cacheon/blob/main/optima/copy_fingerprint.py)
- [Finalized copy reconciliation](https://github.com/latent-to/cacheon/blob/main/optima/chain/intake.py)
- [Validator loop integration](https://github.com/latent-to/cacheon/blob/main/optima/chain/validator_loop.py)
- [Copy behavior tests](https://github.com/latent-to/cacheon/blob/main/tests/test_copy_fingerprint.py)
