# Threat model

Optima evaluates attacker-supplied Python, patches, and native GPU code. The production
design assumes the candidate engine is fully hostile. Static scanning, typed manifests,
and correctness tests reduce exposure; they are not the primary containment boundary.

This page describes implemented controls and the risks they do not close. It applies to
the finalized-intake and OCI qualification path, not to direct development diagnostics.

## Security objective

Optima does not attempt to make candidate code trustworthy. It tries to ensure that a
hostile candidate can influence only the bounded engine computation it was admitted to,
and that no economic or product transition occurs unless trusted, reopenable evidence
proves the registered conditions.

That produces three independent control goals:

1. **Containment:** candidate code cannot reach ambient secrets, networks, state, or later
   workloads through an authorized interface.
2. **Measurement integrity:** the host, not the candidate, owns identity, workload, role,
   timing, output buffers, quality authority, and verdict construction.
3. **Promotion integrity:** crown, settlement, weight publication, integration, signing,
   and serving are distinct authenticated transitions.

A control can satisfy one goal without satisfying the others. No-egress OCI improves
containment but does not make candidate-reported throughput trustworthy. A valid crown
proves registered measurement, not safe production source.

## Protected assets

- Validator wallet and chain-signing authority.
- Model weights and exact model identity.
- Incumbent and candidate proposal bytes.
- Workload selection secrets and hidden tasks.
- Host-owned timing, role assignment, and device observations.
- Raw referee evidence, calibration, and SQLite state.
- Evaluation-stack and reward-family state.
- Release signing keys, reviewed source, registry identity, and serving model.

## Trust boundaries

| Component | Trust position |
|---|---|
| Chain and finalized event history | External consensus/order authority; availability and correctness assumed according to deployment policy |
| HTTPS origin | Untrusted byte transport; URL, routing, archive, and content hash are validated |
| Candidate proposal and native output | Hostile |
| Candidate OCI engine | Hostile process inside constrained runtime |
| Trusted controller and arena provider | Validator authority; must be reviewed and operated securely |
| Pristine T worker | Candidate-free quality authority, still dependent on reviewed runtime/model/reference |
| SQLite and evidence store | Trusted durable state, protected from unprivileged mutation; host administrator remains in scope |
| Weight signer and release signer | High-value control plane, kept outside evaluator containers |
| Reviewed release container | Trusted product artifact after verification; not treated as a hostile candidate sandbox |

## Threats and controls

| Threat | Implemented control | Residual or assumption |
|---|---|---|
| Hostile URL, SSRF, redirect, or archive | HTTPS only; TLS 1.2 minimum; globally routable DNS answers; pinned reviewed IP with SNI/hostname verification; every redirect revalidated; bounded raw gzip/tar preflight including PAX/GNU extension payloads; strict member/size/path rules; committed hash rederived | CA/DNS/origin compromise can affect availability; network stack and TLS library remain trusted |
| Proposal changes after commit | Content hash is checked after extraction and again across immutable publication | Hash does not establish authorship, license, or safety |
| Candidate imports or patches trusted controller | Controller parses candidate as data and never imports candidate Python/native; complete engine runs in a separate OCI worker | A container/kernel/runtime escape can cross the boundary |
| Candidate exfiltrates model/evidence | Runtime has no network, read-only root, exact read-only mounts, private cache tmpfs, bounded protocol; prebuild has no model/GPU/network/home/wallet | GPU/driver side channels, co-tenancy, host compromise, and undiscovered runtime flaws remain possible |
| Candidate tampers with timer or role | Host assigns physical resident lanes and B/C/B′[/C′/B″] roles, owns clocks, validates bounded raw batches/token counts, observes device state, and controls teardown; a reproduction must exchange lane roles | Host clocks, firmware, driver, and provider scheduling must be trustworthy and calibrated |
| Candidate fakes quality | Any required audit is collected in a separate eager/untimed role and host-regraded; candidate speed lifetimes are destroyed before candidate-free pristine T teacher-forces sealed trajectories | Reference bugs, audit sampling limits, and finite hidden-work coverage remain possible |
| Candidate behaves only on known shapes/prompts | Post-commit selection, hidden work, typed graph requirements, and registered decode/long-prefill mixture | Workload overfitting cannot be eliminated; corpora and regimes need ongoing governance |
| Candidate exploits noise | Frozen adaptive B/C/B′ then optional C′/B″ schedule, maximum-noise threshold, `NO_DECISION`, physical-lane-swapped reproduction, and lower reproduced speedup | Hardware drift and cross-validator variance remain operational concerns |
| Candidate hangs or exhausts resources | Stage deadlines, CPU/memory/PID/file/shm/tmpfs bounds, cohort admission, retry budgets, forced container cleanup, durable leases | A GPU/driver hang may require host reset; sustained spam can still consume bounded capacity |
| Candidate persists into later work | Ephemeral containers, read-only mounts, private tmpfs/cache, lease-scoped resources, restart recovery, post-run quiescence checks | Host/container-runtime compromise can persist beyond these controls |
| Copy/front-run attack | Native timelock, finalized event priority, submitted-delta exact/normalized/containment fingerprints | Obfuscated or independently convergent implementations may evade or collide; structural similarity is advisory |
| State replay or partial settlement | Chain-scoped single-writer SQLite, durable cursor/statuses, content-addressed evidence, lease generations, atomic settlement | Disk loss, privileged database edits, and faulty backup/restore are operator risks |
| Weight publication ambiguity | Separate signer; live metagraph refresh; intent-before-submit journal; exact recipient-set plus fixed-tolerance normalized-value readback and `last_update`; held state | Hotkey theft, malicious operator, chain faults, and policy disagreement remain external risks |
| Crown automatically reaches production | Integrated-only release manifest, model seal, deterministic artifacts, SBOM/provenance, Ed25519 signature, expected key, registry reproducibility attestation | Integration review, key custody, base image/toolchain security, registry, and rollout policy remain human/operational authorities |

## Candidate isolation boundary

Production uses two candidate stages:

1. A native **prebuild** container parses the materialized tree and invokes only
   validator-registered build patchers. It has no network, GPU, model, wallet, Docker
   socket, or ambient host mount.
2. A GPU **runtime** container receives only the exact model, materialized engine tree,
   reopened native publication, selected GPUs, private runtime cache, and bounded session
   protocol.

Both use a digest-pinned local image, read-only root, dropped capabilities,
no-new-privileges, reviewed seccomp, non-root UID/GID, resource bounds, private `/tmp`,
and no network. See [Isolation](isolation.md).

## Abuse stories

Concrete attacker stories help reviewers test the composition of controls:

### “I will make stock code look like my acceleration”

The candidate tries to remain ineligible or fail and rely on fallback while the server
still answers. Pre-selection stock routing is allowed for ordinary availability, but
strict qualification requires candidate selection plus positive `fired`/`completed`
coverage and treats selected-path fallback as invalid evidence. End-to-end host timing is
bound to the exact candidate launch identity.

### “I will grade outputs that I know how to game”

The candidate does not choose prompts, selection entropy, role, or quality reference. Its
trajectory is sealed, the candidate is destroyed, and pristine T teacher-forces the
selected trajectory and hidden work. This reduces self-grading and known-prompt attacks;
it does not eliminate finite-workload overfitting.

### “I will persist into the next candidate”

Writes are lease-scoped and ephemeral, roots/mounts are bounded, no network is available,
and the host force-cleans labeled resources then proves device/process quiescence. If it
cannot prove absence, the attempt is not converted into a candidate verdict and the host
must be drained. Kernel/driver escape remains residual risk.

### “I will turn a crown into production code”

Evaluation manifests may name hostile proposals; release manifests cannot. Integration
must preserve the selected payload identity while separately approving provenance,
license, security, compatibility, and tests. The signed release, external expected key,
reproducible registry identity, host authorization, and serve receipts are later gates.

## Control-plane separation

Evaluator workers never receive chain keys. `chain-validate` does not sign weights.
Legacy `set-weights` and active-composition `set-debt-weights` run separately against
durable state and live chain readback. Wallet-free `chain-activate-incentives` binds a
reviewed V2 authority but cannot publish a vector.

Release signing is separate again. A release build context contains a public verification
key, never private signing material. The serving release uses reviewed source and may use
host networking; it is not the hostile candidate container.

## Non-authoritative development paths

`scan` and `verify` are contributor diagnostics. Contributor-controlled matched A/B
profiling may load candidate code in model workers, but its process boundary is not
equivalent to the production OCI controller, evidence graph, or pristine reference. No
contributor-controlled execution path is acceptable for crownable work.

## Explicit nonclaims

Optima does not claim:

- formal verification of Python, CUDA, OCI, the Linux kernel, GPU firmware, or drivers;
- prevention of every side channel or cross-tenant attack;
- complete detection of plagiarism or semantic equivalence;
- immunity to denial of service or evaluator-capacity exhaustion;
- universal performance or quality beyond the registered arena;
- protection against a malicious host administrator, arena provider, reference, or
  release signer;
- Byzantine agreement among validators; or
- that signatures, SBOMs, and reproducible digests replace security and license review.

Security status should be stated as “implemented under these assumptions,” not simply
“closed.”

## Security-review checklist

- Identify whether the change affects intake, prebuild, runtime, reference, settlement,
  signing, or serving; do not reuse controls from a different boundary by analogy.
- List new bytes, mounts, devices, network paths, environment keys, protocol fields,
  patchers, subprocesses, and durable state visible to hostile code.
- Confirm every candidate-controlled value is bounded and authenticated before allocation,
  import, compilation, launch, or persistence.
- Exercise malformed, oversized, timeout, crash, partial-write, restart, and cleanup-race
  paths; verify ambiguous infrastructure cannot mint `FAIL` or `PASS`.
- Prove stock fallback cannot masquerade as candidate execution in strict mode.
- Reopen the resulting evidence and reconstruct the decision without process memory or
  operator narrative.
- Check that evaluator keys, release keys, wallet state, Docker control, model provenance,
  and hidden-work bytes stay outside candidate interfaces.
- State residual risk explicitly, including host root, kernel/container runtime, GPU
  driver/firmware, registry, reference policy, and signing-key custody as applicable.

## Source anchors

- [HTTPS intake](https://github.com/latent-to/cacheon/blob/main/optima/chain/fetch.py)
- [OCI native prebuild](https://github.com/latent-to/cacheon/blob/main/optima/eval/oci_prebuild.py)
- [OCI runtime controller](https://github.com/latent-to/cacheon/blob/main/optima/eval/oci_backend.py)
- [Causal qualification](https://github.com/latent-to/cacheon/blob/main/optima/eval/qualification_runner.py)
- [Weight publication](https://github.com/latent-to/cacheon/blob/main/optima/chain/weights.py)
- [V2 debt publication](https://github.com/latent-to/cacheon/blob/main/optima/chain/debt_publication.py)
- [Release host](https://github.com/latent-to/cacheon/blob/main/optima/release_host.py)
