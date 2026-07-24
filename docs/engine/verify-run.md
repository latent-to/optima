# Verify and run a release

A release consumer needs two trust inputs obtained independently of the release
tree: the expected Ed25519 public key and, for a pinned deployment, the expected
descriptor digest.

Install Optima with its `release` extra before using these commands; it supplies
the required Ed25519 cryptography implementation.

The verification path is intentionally redundant:

```text
externally trusted key + expected descriptor
  -> reopened release publication
  -> deterministic container context
  -> reproducible registry digest
  -> inspected authorized container
  -> verified startup + seam receipts
```

Each step authenticates a different boundary. Skipping from a valid release directory to
`docker run <tag>` drops the registry, host-policy, and activation proofs.

!!! danger "Verification primitives are not rollout authority"
    The commands and APIs below authenticate individual boundaries. Rollout additionally
    requires clean runtime-wheel import, signed native-profile propagation, closed
    effective SGLang arguments and management routes, authenticated builder-output
    binding, and release/session-bound execution receipts. If any product is absent, stop
    before authorization or start. The dated
    [State of record](../reference/state-of-record.md) records the pinned implementation's
    completed and blocked gates.

## Reopen and verify

```bash
python -m optima.cli release-verify /srv/optima/releases/<digest> \
  --expected-public-key <public-key-hex> \
  --descriptor-digest <descriptor-sha256>
```

Verification checks the canonical descriptor and signature, exact top-level
inventory, artifact sizes and hashes, read-only modes, Engine-tree identity,
integrated-only stack and review coverage, native publication, model receipt,
SBOM, provenance, runtime artifacts, and policy manifests. It then computes
the release-tree digest over the verified publication; there is no separate
expected release-tree-digest input.

The command prints the descriptor, Engine-tree, public-key, and release-tree
digests. Retain them with deployment records.

Verification is non-mutating. If it fails, preserve the publication for investigation and
continue serving the previous verified release. Do not chmod files, regenerate a receipt,
or replace the mismatched artifact in place; all of those actions destroy the relationship
to the signed descriptor.

!!! danger "Do not trust an embedded key"
    `release.sig.json` names a public key so signatures can be parsed, but that
    key does not establish its own trust. Always pass the expected public key
    from an independent release channel.

## Materialize the OCI context

```bash
python -m optima.cli release-context /srv/optima/releases/<digest> ./optima-context \
  --expected-public-key <public-key-hex> \
  --descriptor-digest <descriptor-sha256>
```

The destination must not already exist. The command verifies the release again
and emits a frozen deterministic context containing:

- the complete release under `release/`;
- `Dockerfile`;
- `deployment.json`;
- the reviewed `seccomp.json`; and
- `trusted-release-key` containing only the external public verification key.

No chain endpoint, wallet, proposal URL, or private signing material enters the
context.

## Implemented host primitives

The host module exposes these programmatic primitives:

1. `publish_container_twice()` reopens the context, runs two no-cache BuildKit
   builds, pushes them under distinct temporary tags, reopens both Registry-v2
   manifests, requires one raw digest, and signs that common registry identity. A
   production implementation must additionally authenticate that each readback is the
   result emitted by its corresponding builder invocation. The function's current return
   value does not supply that builder-output binding.
2. `authorize_release_container()` binds the signed statement and reopened
   registry image to independently supplied public-key, release-descriptor,
   platform, repository, and image-digest expectations.
3. `create_release_container()` pulls by immutable digest, inspects the local
   image, mounts the exact provisioned model, creates (but does not start) the
   container, and rechecks its complete host policy.
4. `ReleaseContainerHandle.start()` inspects once more immediately before
   start. After the smoke, `copy_receipts()` extracts the bounded receipt
   directory for `verify_serve_receipts()`; `destroy()` force-removes the
   container and volumes.

The following composition is useful for integration testing of those primitives. It is
not a production authorization recipe: before executing it on a release path, deployment
orchestration must add the missing builder-output, clean-wheel, effective-argument,
management-route, native-profile, and receipt-session gates. Values prefixed with
`approved_` must come from independent trust and rollout records; do not derive them from
the registry object being authorized.

```python
from optima.release import verify_serve_receipts
from optima.release_host import (
    RegistryV2Client,
    authorize_release_container,
    create_release_container,
    publish_container_twice,
)

# Integration composition only. The caller must separately authenticate that each
# registry readback came from its corresponding builder output.
signed_build = publish_container_twice(
    context_root,
    repository=repository,
    expected_descriptor_digest=approved_descriptor_digest,
    expected_public_key=trusted_public_key,
    signing_private_key=protected_attestation_private_key,
)

# After the resulting digest is approved through the rollout authority, reopen
# that exact registry object instead of trusting a mutable tag.
registry_image = RegistryV2Client(repository).reopen(
    f"sha256:{approved_image_digest}",
    expected_platform=approved_platform,
)
authorization = authorize_release_container(
    release_root,
    signed_build,
    registry_image,
    repository=repository,
    expected_public_key=trusted_public_key,
    expected_descriptor_digest=approved_descriptor_digest,
    expected_platform=approved_platform,
    expected_image_digest=approved_image_digest,
)

handle = create_release_container(
    authorization,
    provisioned_model_root,
    name=canary_name,
)
try:
    handle.start()
    run_external_health_inference_and_load_smoke()
    receipt_root = handle.copy_receipts(fresh_receipt_destination)
    receipt_proof = verify_serve_receipts(authorization.release, receipt_root)
finally:
    handle.destroy()
```

The code intentionally leaves every missing gate, the smoke function, and rollout
approval to deployment orchestration. `release_host.py` cannot manufacture those
authorities. In particular, the shown `receipt_proof` checks coverage in a directory; it
does not bind the receipt frames to this descriptor/session or check direct-artifact AOT
coverage. If any operation before `try` fails after container creation,
`create_release_container()` performs best-effort cleanup; after it returns a handle, the
caller owns the guaranteed `destroy()` path.

The host contract requires a read-only root, all capabilities dropped,
`no-new-privileges`, reviewed seccomp, a read-only model bind, a 16 GiB
`/tmp` tmpfs (`nosuid,nodev,noexec`), 32 GiB shared memory, host IPC, host
networking, and the exact signed entrypoint/arguments/environment. It requests
all GPUs visible to that host boundary and verifies Docker's `Count=-1`; outer
scheduling must expose only the intended device pool.

Host networking also exposes the server's management surface. Release authorization must
disable or authenticate every management route; successful image/argument inspection
alone does not provide that control.

Unlike the candidate referee, the serving container is intentionally not
no-egress. The host verifier also does not enforce `Config.User` or add
a Dockerfile `USER`; base-image and release review must own that decision. Do
not document the serving boundary as non-root until code enforces it.

## Production startup and rollback contract

Perform this sequence only after the [release acceptance checklist](release-workflow.md#publication-checklist)
and the dated [State of record](../reference/state-of-record.md) show that every required
gate is implemented and retained:

1. Verify the release and context from immutable storage.
2. Produce the reproducible registry identity through the closed host APIs and
   authenticate each registry object against its corresponding builder output.
3. Confirm model content and image digests before scheduling.
4. Create, inspect, and start a canary with the signed arguments and exact host
   policy.
5. Exercise health, inference, and representative quality/throughput probes.
6. Copy the fresh receipt directory and run
   `verify_serve_receipts()` against the reopened release. It requires
   `active`, `fired`, and `completed` coverage for every released slot and TP
   rank, with no `load_failed` or `fallback` receipt. Independently require the input
   frames to bind this release descriptor and serve-session identity. For a release with
   sealed direct artifacts, require per-rank `aot_loaded` and `aot_invoked` coverage as
   well; the current generic verifier does not enforce either additional condition.
7. Promote by immutable descriptor/image identity only after every external and typed
   gate succeeds. Retain the bound receipt product; an unbound
   `ServeReceiptVerification` digest is not promotion authority.
8. Destroy the canary; roll back by selecting a previously verified release,
   never by mutating the
   current tree in place.

!!! note "Operational proof requirement"
    Unit and slice tests do not authorize rollout. The release must retain a clean-wheel
    import/entrypoint proof, provider-specific native authority, authenticated builder
    outputs, closed serving arguments and management routes, a serve smoke at the approved
    topology, and descriptor/session-bound receipts (including AOT coverage when
    applicable). See
    [State of record](../reference/state-of-record.md) for completed evidence.

Serving does not require chain access. A chain outage can delay new proposal
evaluation or attribution updates, but it cannot prevent rebuilding or running
an already signed release.

## Failure triage

| Failure point | Meaning | Safe response |
|---|---|---|
| Signature/public-key mismatch | The publication is not authorized by the expected trust root | Quarantine it and verify key distribution plus descriptor identity |
| Inventory, mode, tree, native, or model mismatch | Supplied bytes are incomplete, changed, or from another release | Recover exact content-addressed inputs; never patch the signed tree |
| Context destination exists | Deterministic emission refuses ambiguous reuse | Verify/delete only through operator-controlled staging policy, then emit to a new absent path |
| Double builds disagree | Container construction is not reproducible under the claimed procedure | Diagnose build inputs/toolchain; publish no authorization statement |
| Registry digest/labels differ at authorization | Registry object is not the attested product | Hold deployment and reopen the registry object by digest |
| Created container fails host-policy inspection | Runtime privileges, mounts, namespaces, command, or environment differ | Destroy before start and correct orchestration policy |
| Startup verification fails | Container contents/model/active seccomp/signed command are wrong | Container exits fail-closed; retain receipts/log identity and roll back |
| Smoke has missing `fired`/`completed` or any fallback | The server did not prove the released paths ran on every expected rank | Do not promote even if health/inference probes answered |
| Device or container cleanup cannot be proven | Operational authority remains ambiguous | Isolate/drain the host before another canary |

## Deployment record

Retain enough information to reproduce the decision, not just the service URL:

- release descriptor, release-tree, Engine-tree, model-content, native-publication, and
  public-key digests;
- raw registry image digest, repository, platform, and signed reproducibility statement;
- authorization result and inspected host-policy identity;
- deployment/canary identity, intended GPU pool, signed command, and model mount;
- health/inference probe results plus the complete fresh serve-receipt verification digest;
- promotion or rollback decision, operator identity, time, and previous release identity.

This record lets an incident reviewer answer both “which bytes were authorized?” and “did
the intended accelerated paths actually run?” without consulting live chain state.

Source: [`optima/release_runtime.py`](https://github.com/latent-to/cacheon/blob/main/optima/release_runtime.py),
[`optima/release_host.py`](https://github.com/latent-to/cacheon/blob/main/optima/release_host.py), and
[`optima/release.py`](https://github.com/latent-to/cacheon/blob/main/optima/release.py).
