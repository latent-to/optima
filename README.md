# Optima

Optima is an inference-throughput competition on SGLang, built to run as a
Bittensor subnet. Miners submit GPU **kernels** (Triton / CuteDSL) targeting
individual operations in a **fixed** model; the validator swaps each kernel into
the model it controls, measures **throughput** under CUDA graphs, and gates the
result on **output fidelity** through validator-owned reference and task checks.
Historical B300 campaigns proved the sampled in-engine comparison against stock.
The causal path now transports bounded raw audit facts from a separate eager,
untimed candidate role and requires a typed, host-regraded witness in every durable
qualification report. Those new bytes are CPU/mock-covered but not yet qualified on
the exact production MiniMax-M3 arena; see [docs/FIDELITY.md](docs/FIDELITY.md).
A kernel earns only if it is faster at equal quality.

The normal competition lane uses registered singleton or atomic kernel targets.
Cross-cutting work can instead enter a fenced reviewed-discovery route; it does
not silently inherit a target identity or automatic reward. This repository
contains the validator harness, chain integration, deterministic Optima Engine
release tooling, and example miner bundles. Untrusted candidate
execution is confined to validator-owned, no-egress OCI workers; the chain and
wallet control plane does not load miner code.

The production path is deliberately staged. Finalized chain arrivals are retained in
SQLite, published to immutable worker trees, and admitted by a registered validator-
owned arena service. Static, build, ABI, graph, and abbreviated-serving screens are
non-crownable. A promoted candidate must pass two independent full qualifications
before transactional settlement; a single pass remains `reproduction_pending`.
Reviewed winners enter a separate chain-independent release path that seals the model
tree and emits signed source, wheel, SBOM, provenance, and OCI build-context artifacts.

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
- **2026-07-18 — the initial V2 economics and discovery composition were
  selected; their historical evidence remains retained.**
  D-012 selected finite log-relative registered-CROWN debt after 224,000
  synthetic runs; D-013 selected a separate 5%-capped, one-epoch, 90-day reviewed
  discovery bounty after 3,240 synthetic rows. Both sweeps replayed byte-identically
  locally and on the RTX pod. D-014's separate 288-row review-delay sensitivity
  also replayed byte-identically across the two architectures; its preregistered
  0/1/7-day SLA passed all 108 rows with 100% discovery payout, no expiry/unissued
  debt, at most 55,555 ppm instantaneous CROWN-capacity dilution, and no CROWN
  paid-fraction regression. The 90/120-day cases issued no stale debt; 30/60/89
  days were diagnostic only. These are synthetic accounting results, not external
  review, activation, publication, durable-state completion, or GPU-performance
  evidence. A signer-free composed shadow then ran against
  testnet netuid 307 at finalized block 7,586,146 (metagraph size 6), mapping
  explicitly synthetic states to 850,000 ppm registered-CROWN, 50,000 ppm
  reviewed-discovery, and 100,000 ppm reserve, totaling 1,000,000 ppm
  (`submitted=false`; semantic digest
  `3dbb3cc27dfd013023c42ba68dd03413d5e5ab1dc8e8626dda3c1a0db18cabaa`,
  file SHA-256
  `ac695810671cdc6f635a9b30a7fb67f1a885e13bd4fba7e64f2456a08ae88aed`).
  A separate restart/cardinality audit made exact settlement speed, principal,
  clocks, lifecycle events, and balance transitions fail-closed; its pre-D-015
  results were 2,135 passed/19 skipped locally and 111 historical conformance tests
  on the pod. D-015 itself has no pod receipt. A fresh live
  intake-only pass then restarted with zero duplicate work. It used no wallet and
  supplies no review, settlement, publication, or activation authority. At that
  checkpoint, legacy V1 remained the sole wired publisher. The selected pure policy's
  promotion-or-bounty rule is not yet end-to-end enforcement: schema 5 retains
  `review_pending` wins and can issue bounded `bounty_only` debt, but rejects
  `registered_promotion` until typed promotion transport, target registration,
  fresh requalification/CROWN linkage, and cross-lane work identity exist. The
  90-day bounty clock starts at the retained win, not the later review. The
  wallet-free atomic one-campaign cutover, schema-6 rollback fence, gapless
  publication/readback/debit path, and causal audit transport are now implemented
  but have no live activation/publication or new-path GPU receipt. Launch still needs
  the exact catalog/reserve manifests and fresh shadow, independent review/runtime-
  invalidation authority, membership-departure history, reliable review-expiry
  scheduling, promotion/cross-lane linkage, the production audit GPU canary, and
  actual operator activation. Its registered-CROWN policy divided capacity by target family;
  D-015 below supersedes that claim-sizing hierarchy, so the old testnet shadows do
  not authorize current policy bytes.
- **2026-07-19 — D-015 selected model-campaign claim sizing.** Claims in the sole
  MiniMax-M3 launch campaign use 100% sizing. Historical two-campaign 50/50 cells
  remain arithmetic research; rotation, a second campaign, and successor activation
  have no live path. Target families remain
  independent frontiers and clocks, but adding 1, 10, or 100 of them causes zero
  principal dilution. All 14 preregistered screens passed. With `k=1`, the normal
  weekly load was one full-sized 4.4%/5% claim for one campaign, or one half-sized
  claim in each of two campaigns (one full share aggregate), rotated across
  families. It paid 100%, expired zero, drained to zero, and had five-day maximum
  latency under empty and saturated discovery; sustained simultaneous per-family
  wins were not the normal-tape assumption. Five-day cadence
  was marginal and four-day cadence overloaded. `k=1.25` was already marginal; at
  `k=1.5` the worst rows overloaded while other rows remained marginal, and `k=2`
  was plainly overloaded. Report digest
  `7975a10b2924330cd527e29b0dfe6f2d9dcb40039f9d8f695b558ec6c6f46590`;
  the raw D-015 sweep is retained in local-only experiment records.
  The one-campaign policy, atomic activation, confirmed publisher/debit path, and
  audit witness transport are implemented in this draft but remain inactive.
  Current local validation is 2,191 passed/19 skipped; the tracked 64-cell D-015
  launch-load replay has semantic digest
  `505fed4d40a6acc6bc92d6330170e8e2260a52e5f3099c22a6c0eb4b2308c672`.
  There is no D-015 pod, activation, publication, or new audit-path GPU receipt.
- Most faithful kernels measure *slower* than the pinned baseline — sglang's
  kernels are heavily tuned. The record above is what passing actually looks
  like: a ~1.04–1.07× measured speedup at zero fidelity violations.

## Docs map

| Doc | What it covers |
|---|---|
| [docs/STATE_OF_RECORD.md](docs/STATE_OF_RECORD.md) | the detailed, numbers-first record — results, calibration, run recipes, ABI, security model. **Wins where docs disagree.** |
| [docs/MINER_GUIDE.md](docs/MINER_GUIDE.md) | the miner on-ramp: gates, slots, bundle anatomy, first kernel in 20 minutes, GPU testing, chain submission |
| [docs/INCENTIVES.md](docs/INCENTIVES.md) | the selected miner rewards: finite log-relative CROWN debt, bounded reviewed discovery debt, examples, and activation status |
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
