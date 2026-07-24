# Documentation standards

Optima's documentation is versioned with the implementation it describes.
`docs/` contains the Markdown source, and `mkdocs.yml` defines the complete
published navigation. Generated `site/` output is never committed.

## Source-of-truth rules

Documentation should identify authority rather than create a second authority.

- Executable behavior comes from the referenced implementation and tests.
- Normative product, slot, manifest, and policy contracts are named explicitly.
- The state-of-record page separates implemented behavior, retained evidence,
  and remaining production work.
- Operator guides describe failure behavior and required trust inputs, not only
  a happy path.
- Historical sequences belong in the design-evolution page. Task and reference
  pages should remain neutral, current, and engineering-focused.

When two pages overlap, choose one canonical explanation and link to it. Do not
copy a contract into several guides; copied prose drifts independently.

## Writing style

- Lead with the invariant, interface, or operational outcome.
- Name the component that owns each decision or piece of evidence.
- State preconditions, outputs, side effects, refusal conditions, and residual
  limitations.
- Use exact command, type, field, and file names.
- Label synthetic, local, GPU, testnet, and production evidence precisely.
- Avoid chronological narration, launch language, and unsupported readiness
  claims outside the dedicated history and state-of-record pages.
- Do not include usernames, home-directory paths, private hosts, wallet
  material, unpublished artifact locations, or agent-session paths.

Examples should use placeholders such as `<network>` and repository-relative
paths. Commands must be safe to copy in the audience and environment stated by
the page.

## Pages and navigation

Every canonical Markdown file under `docs/` is a published page and must appear
exactly once in `mkdocs.yml`. When adding, moving, or removing a page:

1. update the navigation in the same change;
2. update ordinary inbound links instead of leaving compatibility copies;
3. remove the superseded page when its content has a canonical home; and
4. run the repository checker and strict build.

This no-orphan rule keeps old drafts from remaining discoverable but invisible
to maintainers.

A root-level compatibility page is allowed only to preserve an established
documentation URL or a path embedded in a content-addressed release or example
input. It must use the exact `docs-redirect` template enforced by
`scripts/check_docs.py`, contain no independent guidance, and appear in
`not_in_nav`. The checker derives the permitted set from those typed pages and
rejects both redirect chains and a manually growing hidden-page allowlist.

## Links to repository source

Use relative links between documentation pages. Link code on the default branch
with the canonical repository URL:

```text
https://github.com/latent-to/cacheon/blob/main/optima/example.py
https://github.com/latent-to/cacheon/tree/main/examples/example_bundle
```

The documentation checker verifies that each `blob/main` target is a local file
and each `tree/main` target is a local directory. Use a commit-pinned link only
when the historical identity is itself relevant; do not use a commit link for a
current API reference.

## Keeping documentation with code changes

Update the relevant page in the same pull request when code changes any of the
following:

| Change | Documentation normally affected |
|---|---|
| CLI command or option | CLI reference and the task guide that invokes it |
| Manifest, receipt, or wire schema | schema reference, producer, and consumer guides |
| Slot, target, or seam contract | architecture, miner guide, and reference catalog |
| Qualification or settlement behavior | validator guide and state of record |
| Security boundary or refusal | threat model and affected operator guide |
| Environment or compatibility pin | development or compatibility guide |
| Release artifact or verification | Engine and release-operation guides |

Pure refactors and tests may have no documentation impact. Select that
declaration in the pull request template and explain why public behavior and
engineering contracts are unchanged.

Structural checks can detect inventory drift, but they cannot safely generate
semantic documentation from implementation alone. Reviewers still need to
verify that changed behavior, authority, and limitations are described
accurately.

## Local validation

Install documentation dependencies separately from the package runtime:

```bash
python -m pip install -r docs/requirements.txt
```

Run the repository-specific contract checker:

```bash
python scripts/check_docs.py
```

It validates:

- internal file links and Markdown heading anchors;
- complete, duplicate-free MkDocs navigation;
- the documented command inventory against `optima/cli.py`;
- canonical Cacheon `blob/main` and `tree/main` source targets;
- retired repository URLs; and
- host-private and agent-private paths.

Then render with warnings treated as errors:

```bash
python -m mkdocs build --strict
```

For an interactive preview:

```bash
python -m mkdocs serve
```

The same checks run in GitHub Actions. Pull requests build the site without
deploying it; a successful push to `main` publishes the GitHub Pages artifact.
