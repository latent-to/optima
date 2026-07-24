# Model provisioning

Model provisioning seals exact model bytes into a path-independent,
content-addressed receipt before release construction. It proves which regular
files existed; it does not assign arena or serving policy.

Provisioning should be understood as taking an inventory under a microscope, not as
downloading or blessing a model. The operator has already selected and placed a concrete
model revision. Optima turns that tree into an immutable identity which release
construction and startup can later reopen.

## Provision a tree

Both arguments must already be concrete directories. Create the publication
directory outside the model tree before running the command:

```bash
install -d -m 0700 /srv/optima/model-publication

python -m optima.cli model-provision \
  /srv/models/model-revision \
  /srv/optima/model-publication \
  --expected-content-digest <sha256>
```

The command recursively inventories standalone regular files in canonical path
order, hashes stable bytes, rejects transient `.cache` paths, symlinks,
hard-linked files, and special objects, detects tree changes during the scan,
and writes a content-addressed receipt outside the model tree.
It prints:

- `content_digest` — identity of the complete path/size/hash inventory;
- `receipt_digest` — identity of the canonical receipt itself;
- `receipt_path` — immutable published receipt location.

`--workers` controls hashing concurrency and defaults to four. Supplying
`--expected-content-digest` turns a previously established model identity into
a hard precondition.

## Receipt semantics

Each receipt is exact-schema JSON with:

```json
{
  "type": "optima.model-provision",
  "schema_version": 1,
  "content_digest": "<sha256>",
  "files": [
    {"path": "config.json", "size": 123, "sha256": "<sha256>"}
  ]
}
```

Paths are canonical relative POSIX names. They are sorted and must be unique
both literally and under case folding. The receipt is independent of the host
path where the model was provisioned.

### What changes the identity?

Assume the same model is copied from `/srv/models/rev-a` to `/mnt/models/rev-a` without
changing a byte. Both trees produce the same content digest because the host root is not
part of identity. These changes do produce a new identity or rejection:

| Change | Result |
|---|---|
| Move an unchanged complete tree to another host path | Same content digest |
| Change one weight byte, config byte, filename, or relative path | New content digest |
| Add any path component named `.cache` | Rejected as transient cache state |
| Add another regular lock file or downloader sidecar | New content digest; keep such state outside the root |
| Replace a regular file with a symlink or hard link | Rejected |
| Mutate a file while the inventory is being read | Rejected as an unstable tree |
| Put the published receipt inside the model root | Self-changing input; operationally invalid |

The receipt does not say that the model is accurate, licensed, safe, or compatible with a
particular release. Those decisions belong to model governance and the release descriptor.

## Operational rules

- Treat model content as immutable while provisioning and serving. A changed
  byte is a new model identity.
- Keep the publication root separate from the source model tree. A receipt
  inside the tree would change the content it describes.
- Retain the exact model tree or an independently verifiable content-addressed
  store. The receipt proves identity, not future availability.
- Do not place credentials, downloader state, caches, sockets, or mutable lock
  files inside the model root.
- Bind model ID, revision, manifest digest, content digest, and receipt digest
  into the release descriptor. A human model name is not enough.

Provisioning is intentionally independent of chain state. A release consumer
can verify the model identity and mount the exact bytes without contacting the
proposal market.

## Failure and recovery

- If `--expected-content-digest` differs, stop. Determine whether the source revision,
  download, extraction, or local mutation is wrong; do not update the expected value just
  to make the command pass.
- If the tree changes during scanning, quiesce the downloader or copy the revision into a
  sealed staging tree and provision again.
- If an unsafe filesystem object is found, rebuild the tree from standalone regular files.
  Dereferencing links in place can accidentally change both identity and trust scope.
- If a receipt exists at the content address, reopening must prove it is identical. A
  conflicting existing object is evidence of storage corruption, not an overwrite case.
- If the exact model bytes are lost, the receipt alone cannot restore them. Recover an
  independently retained copy and verify it against the receipt before serving.

## Provisioning checklist

1. Resolve a concrete upstream model revision and record its external provenance.
2. Finish all download/conversion work; remove caches and transient state from the tree.
3. Create an owner-controlled publication root outside the model directory.
4. Provision with the previously approved expected digest when one exists.
5. Retain both the receipt and an independently available exact model tree.
6. Bind model ID, revision, manifest, content, and receipt digests in release inputs.
7. At deployment, mount the tree read-only and let release startup reopen it; do not rely
   on the directory name as identity.

Source: [`optima/model_provision.py`](https://github.com/latent-to/cacheon/blob/main/optima/model_provision.py).
