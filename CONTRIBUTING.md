# Contributing to Optima

Optima combines contribution tooling, validator-controlled evaluation, chain
state, settlement, and serving-release construction. A small change can cross a
security or authority boundary, so contributions should be narrow, testable,
and explicit about the contracts they affect.

## Development setup

Use Python 3.11 or newer for local development. The CPU extra provides the
un-pinned Torch dependency used by contract tests; GPU environments should
install the Torch/SGLang build that matches their CUDA stack first.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[cpu,dev]"
```

Run a focused test while developing, then the relevant broader suite before
opening a pull request:

```bash
python -m pytest -q tests/test_bundle_hash.py
python -m pytest -q tests
```

GPU validation has additional pinned-environment and evidence requirements. See
the [development environment guide](docs/dev/environment.md) before interpreting
a GPU result as project evidence.

## Change discipline

- Keep a pull request focused on one coherent behavior or contract.
- Reuse the existing typed authority boundary instead of adding an alternate
  source of truth.
- Preserve fail-closed behavior at candidate-code, evidence, settlement,
  signing, and release boundaries.
- Add regression tests for behavior changes and negative tests for refusals.
- Distinguish local tests, synthetic evidence, GPU measurements, and production
  operations. One must not be described as proof of another.
- Never commit credentials, wallet material, private endpoints, host-specific
  paths, generated runtime state, or unpublished experiment logs.

Changes to slot or target contracts, candidate execution, evidence authority,
chain publication, economics, or release construction deserve an especially
small diff and an explicit statement of the invariant being preserved.

## Documentation

Documentation lives in this repository so behavior and its engineering contract
can be reviewed in the same pull request. Update documentation when a change
affects a public command, API or schema, operator procedure, security boundary,
failure mode, compatibility constraint, or stated implementation status.

Not every code change needs prose changes. Every pull request must nevertheless
select one documentation-impact declaration in the pull request template:

- documentation updated in the same pull request; or
- no documentation impact, with a short explanation.

Automation catches broken links, missing navigation, stale CLI inventory,
machine-private paths, retired repository URLs, and missing source targets. It
cannot determine whether prose accurately describes a new engineering
behavior; that remains part of code review.

Install the isolated documentation toolchain and run both checks:

```bash
python -m pip install -r docs/requirements.txt
python scripts/check_docs.py
python -m mkdocs build --strict
```

The detailed writing, navigation, and source-link rules are in the
[documentation standards](docs/contributing/documentation.md).

## Pull requests

Before requesting review:

1. Rebase or merge the current target branch and resolve conflicts deliberately.
2. Run the smallest relevant tests plus any required broader suite.
3. Run the documentation checks when Markdown, CLI inventory, or `mkdocs.yml`
   changes.
4. Complete every applicable pull request template section, including the
   documentation-impact declaration.
5. Describe what was tested and what was not tested. Do not promote an unrun
   check or an older artifact as evidence for the current source.

Generated `site/` output, local databases, model bytes, and build artifacts do
not belong in a pull request.

## License

By contributing repository code or documentation, you agree that it is
provided under the repository's [Apache-2.0 license](LICENSE). Miner submission
terms are a separate policy surface and are not changed by ordinary repository
contributions.
