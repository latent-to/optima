# Integrating a contribution

Integration promotes one reproduced crown into ordinary reviewed Optima source.
It is the bridge between adversarial measurement and release engineering, not
an automatic consequence of settlement.

The integration reviewer is not re-running the economic contest or granting a crown.
They are answering a narrower product question: can the exact selected delta that won be
owned, understood, maintained, composed with the reviewed release baseline, and distributed
under an explicit provenance and security record?

## Inputs

Begin only with retained, reopenable authority:

- the proposal and target-spec identities;
- settlement candidate, settlement evidence, and crown-event digests;
- two distinct full qualification attempts;
- the selected target payload and attribution identity;
- the exact target-spec digest under which the contribution was crowned.

Reopen both attempts and the settlement transition. Human-readable summaries
or a chain row alone are insufficient. The complete target-catalog snapshot is
bound later by the `EngineReleaseManifest`; it is not a field of the
integration review record.

## Review sequence

1. **Reproduce against the release branch.** Re-establish the contribution's
   behavior against both its crowned evaluation stack and the current
   integrated Engine stack. Resolve interactions with other active targets.
2. **Promote reviewed source without changing the selected delta.** Place the
   contribution in an ordinary Optima-owned source tree at a reviewed commit.
   The selected payload digest must still match the crowned proposal
   byte-for-byte. Packaging outside that selected closure may be normalized,
   but rewriting the attributable payload requires new evidence rather than
   being hidden inside integration.
3. **Review provenance and license.** Establish the right to distribute every
   source and generated/native input. Preserve contributor attribution as an
   immutable digest-bound record.
4. **Review security.** Treat the submitted implementation as a lead, not as
   trusted source. Review memory safety, input bounds, build behavior,
   dependency patches, graph behavior, failure paths, and fallback behavior.
5. **Check compatibility.** Run the SGLang seam canary and the target-specific
   suite against the pinned runtime, supported hardware, model profile, and
   active stack composition.
6. **Add maintained tests.** Carry component correctness, graph replay,
   end-to-end quality, regression, and negative/fallback cases into the normal
   source tree.
7. **Record approval.** Bind every evidence artifact, the exact source-tree and
   selected-payload digests, reviewer identity, and full review commit in an
   `IntegrationReviewRecord`.

### Worked example

Suppose a reproduced crown names a selected RMSNorm payload digest `P` and target-spec
digest `T`. Integration may move the bundle into a reviewed repository directory, rename
surrounding modules, add normal packaging metadata, and add maintained regression tests.
It may not silently optimize or rewrite the selected payload bytes represented by `P`.

The reviewer therefore proves two things at once:

```text
economic continuity:  crown.selected_payload_digest == review.selected_payload_digest
product identity:     reviewed full tree + commit + evidence -> integrated reference
```

If review finds a security bug inside the selected payload and fixing it changes `P`, the
safe result is not to bless the changed bytes under the old evidence. Keep the crown as a
historical/economic fact, produce a new attributable candidate or other explicitly
authorized evidence for the changed payload, and integrate only after that identity is
resolved. By contrast, changing documentation or packaging outside the selected closure
can be compatible with the preserved crown binding when the full reviewed tree and commit
are recorded.

## Integration record

The record is chain-independent after construction but preserves chain and
crown identities as provenance. It requires distinct retained artifacts for:

- primary qualification;
- independent reproduction;
- license evidence;
- provenance evidence;
- security review;
- compatibility evidence; and
- test evidence.

An approved record derives one `IntegratedContributionRef`. Its target ID,
target-spec digest, integrated source-tree digest, selected-payload digest,
attribution digest, and review-record digest all travel into the release
manifest.

### Who authorizes what

| Actor or product | Authority | Does not authorize |
|---|---|---|
| Two qualification attempts + settlement | The exact crowned target delta and its economic evidence | Shipping, license, maintenance, or security fitness |
| Integration reviewer | Reviewed source, provenance/license/security/compatibility/test evidence | Release signing or deployment |
| `IntegrationReviewRecord` | Immutable binding from crown evidence to reviewed source and attribution | Inclusion in every future release |
| Release manifest owner | Selection of active integrated refs for one product composition | Changing the reviewed source behind a ref |
| Release signer | Authorization of one exact descriptor and closed artifact set | Trust in a key obtained only from that same release |

## Release admission checks

An `EngineReleaseManifest` accepts only integrated references and validates
complete one-to-one coverage with the supplied review records. Materialization
then reopens each integrated source tree, selects only the registered target
payload, rewrites bundle-local module identities, applies validator-owned
rebuild policy, and emits a deterministic Engine tree with read-only regular
files. Its digest and reopen checks, not directory permissions alone, make
mutation detectable.

Reject integration when:

- either qualification attempt is missing, reused, or no longer reopenable;
- source or attribution does not match the selected crowned delta;
- license, provenance, security, compatibility, or tests remain conditional;
- the target conflicts with the active catalog composition;
- the pinned runtime or release model cannot exercise the contribution safely;
- maintained fallback and rollback behavior are absent.

Deferring integration does not invalidate a crown or its attribution. It simply
keeps unreviewed content out of the serving product.

## Review outcome matrix

| Finding | Correct outcome |
|---|---|
| Evidence is missing or cannot reopen | Block; repair retention/authority, not the review record |
| Selected payload differs from the crowned digest | New evidence/identity is required |
| Packaging changes only outside selected closure | Permissible if fully reviewed and bound in the integrated source tree |
| Contribution is sound but conflicts with the current active stack | Defer, redesign composition policy, or target a later release train |
| Provenance or license remains ambiguous | Block distribution; a valid crown does not cure rights uncertainty |
| Fallback hides a selected-path failure | Block until maintained strict-path behavior and tests exist |
| Review succeeds | Derive an integrated ref; inclusion still requires an explicit release decision |

### Reviewer completion checklist

- Reopen both passes and settlement from content-addressed authority.
- Recompute the selected-payload binding instead of copying a digest from a summary.
- Review the complete build closure, including CUDA, includes, vendored files, dependency
  patches, generated inputs, and native outputs.
- Test eager and graph paths, supported topologies, negative inputs, fallback, and rollback.
- Record exact reviewer, commit, tree, target-spec, attribution, and evidence identities.
- Confirm that the resulting ref can be materialized without a miner URL or chain query.
- Leave the crown intact but the release unchanged until a separate product selection.

Source: [`optima/stack_manifest.py`](https://github.com/latent-to/cacheon/blob/main/optima/stack_manifest.py) and
[`optima/engine_tree.py`](https://github.com/latent-to/cacheon/blob/main/optima/engine_tree.py).
