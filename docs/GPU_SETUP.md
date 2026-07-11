# GPU setup — provider-agnostic checklist

How to turn any rented (or owned) CUDA box into a machine that can run Optima's
GPU gates: `verify --device cuda`, `evaluate`, `bench`, and the validator loop.
No specific provider is assumed. (The maintainers' own pod notes, with
provider-specific commands, live in [DEV_ENVIRONMENT.md](DEV_ENVIRONMENT.md) —
you don't need them.)

## 1. What the box needs

- An NVIDIA GPU with recent drivers. Validated so far: H100 (arch `9.0`) and
  B200/B300-class (arch `10.0`/`10.3`). A smaller GPU works for small-model
  smoke runs.
- CUDA toolkit on disk with `nvcc` (sglang JIT-compiles kernels at runtime), and
  `ninja` on PATH.
- Python 3.12 (the validated toolchain) and [uv](https://docs.astral.sh/uv/)
  (fast installer; plain pip works too).
- Disk for the model weights you plan to score against.

## 2. Install

```bash
git clone https://github.com/latent-to/optima && cd optima
uv venv --python 3.12 .venv && source .venv/bin/activate

# sglang brings the matching torch. Install the version validators score against —
# read the pin (and its validation state) from the source of truth:
grep "^PINNED_SGLANG" optima/compat.py
uv pip install "sglang==<the pinned version>" ninja datasets
uv pip install -e .          # no [cpu] extra here — torch came with sglang

# install the seam in EVERY interpreter of this venv (sglang runs the model in a
# spawned child process; parent-process patching never reaches it):
SP=$(python -c 'import site;print(site.getsitepackages()[0])')
echo 'import optima.bootstrap' > "$SP/optima.pth"
```

Known resolver gotchas on CUDA 13 boxes (cost us rebuilds; see
DEV_ENVIRONMENT.md for the full story): recent sglang may need
`--prerelease=allow` (a beta flash-attn dependency) and
`--torch-backend=cu130` to route torch to the right index; if `import sglang`
fails inside `transformers` with a `LayerRepository` error, pin
`"kernels>=0.12,<0.13"`.

## 3. Environment variables

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PWD/.venv/bin:$PATH   # nvcc + ninja for the JIT
export TORCH_CUDA_ARCH_LIST=9.0                        # your GPU arch: 9.0=H100, 10.0=B200
```

Seam arming for block/collective slots is per-run (`OPTIMA_ATTENTION_SEAM=1`,
`OPTIMA_MOE_SEAM=1`, `OPTIMA_COLLECTIVE_SEAM=1`, `OPTIMA_ARFUSION_SEAM=1`) — the
eval harness sets what it needs; you only set these when driving a seam by hand.

## 4. Self-checks (in order, before trusting any number)

```bash
# 1) the seam chokepoints exist in the installed sglang (static, no GPU)
python -m optima.cli compat

# 2) op-correctness on device — a faithful kernel passes
python -m optima.cli verify examples/miner_silu_triton --device cuda

# 3) end-to-end smoke on a small model — and the seam actually fires
python -m optima.cli evaluate examples/miner_silu_triton \
    --model Qwen/Qwen2.5-0.5B-Instruct --no-deterministic
```

The eval demands `active` receipts from every expected scheduler member, then one
`completed` receipt for every registered slot/member pair and zero `fallback`
receipts. `fired` means only that routing selected a candidate; later adapter or
kernel work may still fail or intentionally serve stock. Missing completion aborts
instead of scoring stock-vs-stock. These process-local receipts are diagnostics,
not correctness or isolation authority (see [MINER_GUIDE.md](MINER_GUIDE.md)).

Always launch GPU evals via `python -m optima.cli ...` — sglang spawns the
scheduler with `mp spawn`, so the `__main__` guard matters.

## 5. Multi-GPU (TP) notes

- `--tp-size N` on `evaluate`/`bench`; collective slots verify distributed
  (`optima verify --world-size N` — use the arena's TP size).
- gpt-oss-120b fits a single H100; multi-GPU runs pick a MoE backend via
  `--moe-runner-backend` (see the arena recipes in [STATE_OF_RECORD.md](STATE_OF_RECORD.md)).

## 6. Validator operators

The GPU box runs evaluation only. Chain keys (wallet, weights) belong on a
separate CPU machine — the GPU box executes untrusted miner code and must never
hold secrets. The chain loop, its evaluator contract, and the tested runbook are
in [TESTNET.md](TESTNET.md). Isolation hardening for genuinely hostile bundles
(no-egress namespaces, per-eval wipe) is documented there as an open
requirement — do not point a production wallet at this before reading it.
