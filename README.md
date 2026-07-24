# Optima

Optima is an inference-throughput competition built around SGLang and designed
to operate as a Bittensor subnet. Miners submit inspectable GPU-kernel
contributions at registered boundaries in a validator-owned model. Validators
admit those contributions through a typed, isolated pipeline and reward only
improvements that reproduce under the registered throughput and quality policy.

This repository—published as [`latent-to/cacheon`](https://github.com/latent-to/cacheon)—contains
the miner SDK and examples, validator and chain control plane, evaluation
runtime, settlement and incentive machinery, and chain-independent engine
release tooling.

> [!IMPORTANT]
> Optima is pre-release software. Implemented paths, retained empirical evidence,
> and production readiness are separate claims. The
> [state of record](docs/reference/state-of-record.md) identifies what has been
> exercised, what is covered only by tests, and what remains unproven.

## Start here

| Goal | Documentation |
|---|---|
| Understand the system | [Concepts](docs/get-started/concepts.md) and [architecture overview](docs/architecture/overview.md) |
| Build a miner contribution | [Miner guide](docs/miner-guide/overview.md) |
| Operate a validator | [Validator guide](docs/validator-guide/overview.md) |
| Integrate an approved contribution | [Optima Engine](docs/engine/overview.md) |
| Review trust boundaries | [Security model](docs/security/threat-model.md) |
| Contribute to the repository | [Contributing](CONTRIBUTING.md) |

The rendered documentation is published from this repository at
[latent-to.github.io/cacheon](https://latent-to.github.io/cacheon/).

## Local correctness loop

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[cpu,dev]"

python -m optima.cli slots
python -m optima.cli scan examples/miner_silu_torch
python -m optima.cli verify examples/miner_silu_torch \
  --device cpu \
  --dtype float32
```

`scan` and `verify` are development checks. They do not establish serving
throughput, end-to-end quality, settlement eligibility, or a production
release. Those decisions belong to the validator-owned qualification and
release paths described in the documentation.

## Design boundaries

- The validator owns the model, workload, references, timing, outputs, and
  reward policy. A miner contribution owns only its registered target.
- Candidate build and execution run in validator-owned, no-egress OCI workers;
  wallet and chain-signing authority remain outside candidate lifetimes.
- A single passing qualification is not a crown. Settlement requires an
  independently reproduced pair bound to the same contribution and evaluation
  context.
- Evaluation acceptance and serving release are different decisions. Engine
  releases contain reviewed, integrated artifacts and do not include chain,
  wallet, intake, or settlement code.

See the [product model](docs/architecture/product-model.md) and
[slot contract](docs/architecture/slot-contract.md) for the normative
invariants.

## Tests

```bash
python -m pytest -q tests
```

Documentation checks are described in
[Contributing](CONTRIBUTING.md#documentation).

## License

The repository is licensed under [Apache-2.0](LICENSE). Miner submissions are
governed separately by the
[draft submission terms](docs/legal/submission-terms.md); those terms require
legal review before production use.
