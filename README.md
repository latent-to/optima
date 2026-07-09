# Optima

Optima is an inference-throughput competition on SGLang, built to run as a
Bittensor subnet. Miners submit GPU **kernels** (Triton / CuteDSL) targeting
individual operations in a **fixed** model; the validator swaps each kernel into
the model it controls, measures **throughput** under CUDA graphs, and gates the
result on **output fidelity** — a sampled in-engine comparison against the stock
baseline, plus task accuracy on real benchmarks. A kernel earns only if it is
faster at equal quality.

This repository is the validator harness (the referee), the chain integration,
and example miner bundles.

| You are | Start here |
|---|---|
| writing kernels to compete (**miner**) | [docs/MINER_GUIDE.md](docs/MINER_GUIDE.md) — the slots, the gates, the bundle format, local → GPU testing, submission. No prior subnet knowledge assumed. |
| operating a **validator** | [docs/TESTNET.md](docs/TESTNET.md) (the chain loop runbook) + [docs/GPU_SETUP.md](docs/GPU_SETUP.md) (the GPU box) |
| reading the **design** | [docs/HOW_OPTIMA_WORKS.md](docs/HOW_OPTIMA_WORKS.md) (end-to-end mechanism + threat model) and [docs/SLOT_CONTRACT.md](docs/SLOT_CONTRACT.md) (the four invariants) |

## Quickstart (CPU — no GPU required)

```bash
python3 -m venv .venv && source .venv/bin/activate   # python -m venv ships pip; `uv venv` does NOT
pip install -e ".[cpu,dev]"
python -m optima.cli slots        # the catalog of slots a kernel can target
python -m optima.cli verify examples/miner_silu_torch        --device cpu --dtype float32   # passes
python -m optima.cli verify examples/miner_silu_broken_torch --device cpu --dtype float32   # fails: drops SiLU
pytest tests/
```

That is the local inner loop: copy an example bundle, edit the kernel,
re-`verify`. CPU verify checks op-correctness only; throughput and fidelity are
scored on GPU — setup in [docs/GPU_SETUP.md](docs/GPU_SETUP.md).

## Measured record

- **2026-07-07 — first submissions through every gate.** Two fused-epilogue
  collective kernels measured **1.044×/1.049× (shallow) and 1.074×/1.071×
  (deep)** against the noise-derived bar on the MiniMax-M3-NVFP4 / 4×B300
  arena — full gate chain green, in-engine audit ~12,500 sampled calls /
  0 violations each, reproduced on independent prompt seeds.
- **2026-07-08 — the full loop ran on the public Bittensor testnet** (netuid
  307): timelock commit-reveal → hash-verified fetch → copy fingerprinting →
  the GPU referee, no human in the path. The deep bundle scored **1.072× (bar
  1.026; audit 12,824 / 0)** — its third independent reproduction.
- Most faithful kernels measure *slower* than the pinned baseline — sglang's
  kernels are heavily tuned. The record above is what passing actually looks
  like: a ~1.04–1.07× measured speedup at zero fidelity violations.

## Docs map

| Doc | What it covers |
|---|---|
| [docs/STATE_OF_RECORD.md](docs/STATE_OF_RECORD.md) | the detailed, numbers-first record — results, calibration, run recipes, ABI, security model. **Wins where docs disagree.** |
| [docs/MINER_GUIDE.md](docs/MINER_GUIDE.md) | the miner on-ramp: gates, slots, bundle anatomy, first kernel in 20 minutes, GPU testing, chain submission |
| [docs/HOW_OPTIMA_WORKS.md](docs/HOW_OPTIMA_WORKS.md) | the full design: pipeline, seam mechanism, threat model |
| [docs/SLOT_CONTRACT.md](docs/SLOT_CONTRACT.md) | the four invariants every slot must satisfy (short; read before touching `optima/slots.py`) |
| [docs/FIDELITY.md](docs/FIDELITY.md) | the two fidelity modes (in-engine audit vs rollout-KL), the measured post-mortem, the adversarial matrix |
| [docs/TESTNET.md](docs/TESTNET.md) | validator runbook: register, run the chain loop, evaluator contract |
| [docs/GPU_SETUP.md](docs/GPU_SETUP.md) | provider-agnostic GPU box setup: toolchain, seam install, self-checks |
| [docs/SGLANG_TRACKING.md](docs/SGLANG_TRACKING.md) | how the pinned sglang is bumped and re-baselined (consensus) |
| [docs/SUBNET_BLUEPRINT.md](docs/SUBNET_BLUEPRINT.md) | how a production subnet is assembled (studied from Affine), mapped onto Optima |
| [docs/SUBMISSION_MODEL.md](docs/SUBMISSION_MODEL.md) | advanced: the override submission tier (`base_kernel` / `override_point`) |
| [docs/DEV_ENVIRONMENT.md](docs/DEV_ENVIRONMENT.md) | the maintainers' own dev-pod notes (provider-specific; not required reading) |
| [docs/SUBMISSION_TERMS.md](docs/SUBMISSION_TERMS.md) | **draft** miner submission terms (operator license grant, emissions as sole compensation) — needs counsel review before mainnet |

## License

The harness is [Apache-2.0](LICENSE). Miner submissions to the subnet are
accepted under separate [submission terms](docs/SUBMISSION_TERMS.md) (draft) —
the repo license covers this code, not what miners submit.
