# Dev environment — GPU pods, toolchain, run recipe

How to get a GPU, push the harness, and run evals. Written so a fresh agent can
pick up cold.

## The toolchain env (`sn120`)

A pyenv on the dev macbook has everything for chain + pods + Affine in one place:

```bash
pyenv activate sn120          # ~/.pyenv/versions/sn120  (Python 3.11)
# has: bittensor 10.x, bittensor-cli, bittensor-wallet, lium-cli 0.6, affine
```

If `pyenv activate` isn't wired into the shell, call binaries directly, e.g.
`~/.pyenv/versions/sn120/bin/lium ...` and `~/.pyenv/versions/sn120/bin/python ...`.

This is also where the **Bittensor SDK** lives for when we build the chain layer
(read commitments, set weights). The Affine subnet code is at
`~/Downloads/github/affine` (studied in `docs/SUBNET_BLUEPRINT.md`).

## The GPU pods (lium.io)

We rent GPUs on [lium](https://github.com/Datura-ai/lium-cli) (Datura-ai). The CLI
is configured (`~/.lium`). **Pod names/IPs change on redeploy — always run
`lium ps` to get the current ones.** As of this writing:

| Pod (name) | GPU | CUDA | Access | Notes |
|---|---|---|---|---|
| `golden-lion-b6` | H100 (80 GB HBM) | 13.0 | `216.81.245.218:40309` | where the harness was validated |

Rent a multi-GPU box (e.g. B200) per experiment when you need TP / PD-disagg / EP;
names/IPs are whatever `lium ps` shows.

Some rented pods do not grant `CAP_SYS_ADMIN`; direct `unshare(CLONE_NEWNET)`
therefore fails with `EPERM`. Production qualification does not rely on that direct
path: the validator-owned OCI process manager launches the candidate worker with no
network, a read-only root filesystem, bounded mounts, seccomp/resource policy, and
controller-owned teardown. `--allow-unsafe-no-isolation` is restricted to non-crownable
development replication.

RTX Blackwell hardware is sufficient for referee integration, OCI lifecycle,
distributed graph, testnet, and chain-independent release validation. B300 runs are
reserved for properties the RTX topology cannot establish: SM103/CuTe support,
NVLink/P2P and custom all-reduce behavior, B300-specific noise/calibration, TP4 role
swaps, and performance qualification of the MiniMax-M3 campaign kernels.

### Driving a pod programmatically

All from the `sn120` env (`lium` on PATH there):

```bash
lium ps                                   # list pods + names + IPs
lium exec <pod> "nvidia-smi -L"  # run a command non-interactively
lium ssh  <pod>                  # interactive shell
lium rsync <pod> ./optima        # push a directory (use for the harness)
lium scp  <pod> ./file /root/    # copy a single file
lium logs <pod>                  # stream logs
lium rm   <pod>                  # TERMINATE (stops billing)
```

The H100 also answers direct ssh: `ssh root@216.81.245.218 -p 40309 -i ~/.ssh/id_ed25519`.

> Billing: the H100 is ~$1.48/h; multi-GPU boxes cost more. **Tear down idle pods
> with `lium rm <name>`** — they bill while RUNNING.

## Bootstrapping the harness on a fresh pod

The validated recipe (from the H100; adjust `TORCH_CUDA_ARCH_LIST` per GPU):

```bash
# on your machine: push the harness (NOT the sglang clone)
lium rsync <pod> ~/Downloads/github/optima/optima
lium rsync <pod> ~/Downloads/github/optima/examples

# on the pod (lium ssh, or wrap each in `lium exec <pod> "..."`):
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /root/optima && uv venv --python 3.12 .venv && source .venv/bin/activate
# Latest stable sglang on CUDA 13. --prerelease is needed (0.5.12 depends on the
# flash-attn-4 beta, a pure-python wheel); --torch-backend routes the torch family to
# the cu130 index. Then PIN kernels<0.13 — transformers 5.6 breaks against kernels
# 0.15 ("Either a revision or a version must be specified").
uv pip install --prerelease=allow --torch-backend=cu130 "sglang==0.5.12.post1"
uv pip install "kernels>=0.12,<0.13" datasets pytest -e .
SP=$(python -c 'import site;print(site.getsitepackages()[0])')
echo 'import optima.bootstrap' > "$SP/optima.pth"     # install the seam everywhere

export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PWD/.venv/bin:$PATH   # sglang JIT needs nvcc + ninja
export TORCH_CUDA_ARCH_LIST=9.0                        # set per GPU: 9.0 = H100, 10.0 = B200
.venv/bin/python -m optima.cli verify   examples/miner_silu_triton --device cuda
```

## Clean-signal eval settings

When measuring kernel fidelity, run with `enable_deterministic_inference` so the
nondeterminism noise floor → ~0, and calibrate the KL threshold to a measured
stock-vs-stock floor (see `README.md` "Calibration findings"). Measure the floor
with `optima/.../noise_floor`-style stock-vs-stock runs before trusting a KL number.

## Session log

> Newest first. A running record of what each agent/session actually did, so the
> next one can resume cold. Candid and concrete — commands, numbers, gotchas.

### 2026-06-01 — block slots + attention seam + cu13 / sglang-0.5.12 bring-up (Opus 4.8)

**Box:** the 1×H100 (80 GB, driver 580 / CUDA 13.0) at `ssh root@216.81.245.218 -p 40299`
(note: not the 40309 port in the pod table above — pod ports rotate). Built a fresh `uv`
venv from scratch around the *latest stable* sglang.

**The cu13 env recipe that works (and the gotchas that cost rebuilds):**

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install --prerelease=allow --torch-backend=cu130 "sglang==0.5.12.post1"
uv pip install "kernels>=0.12,<0.13" datasets pytest -e .
echo 'import optima.bootstrap' > "$(python -c 'import site;print(site.getsitepackages()[0])')/optima.pth"
```

Resolves to torch 2.11.0+cu130, flashinfer-python 0.6.11.post1, sgl_kernel, transformers 5.6.
- (1) sglang 0.5.12 **hard-depends on the `flash-attn-4` beta** (a pure-python wheel) → need
  `--prerelease=allow`, else uv refuses and torch never installs.
- (2) Route the torch family to the cu130 index with `--torch-backend=cu130`.
- (3) **Pin `kernels<0.13`** — transformers 5.6 breaks against kernels 0.15 with
  `LayerRepository: Either a revision or a version must be specified` at `import sglang`.

**Shipped — branch `feat/block-slots-attention-seam` (3 code commits + this log):**

1. `feat(slots)`: SlotSpec gains `kind` (op|block), a `Correctness` policy (allclose |
   `matched_ratio` vs high-precision ground truth), and multi-output. Added the `attention.sdpa`
   and `attention.decode` **block** slots + the decode seam: intercept `RadixAttention.forward`,
   `set_kv_buffer` the new token (validator owns the write), gather the paged KV via
   `req_to_token[req_pool_indices][:, :max_len]` + `seq_lens`, run the miner kernel. **Eager-only
   gather MVP** (a per-step `max_len` isn't CUDA-graph-capturable).
2. `fix(sglang)`: 0.5.12 compat. `sandbox.load_entry` now registers the kernel in `sys.modules`
   *before* exec — else 0.5.12's piecewise-CUDA-graph / torch.compile path crashes the candidate
   launch with `ModuleNotFoundError: optima_kernel_*`. Seam installers guard on the class
   attribute instead of a raising import (kills the circular-import traceback on every
   `import sglang`). Compat canary covers RadixAttention; `PINNED_SGLANG 0.5.9 → 0.5.12.post1`.
3. `feat(eval)`: default to **CUDA graphs ON + auto attention backend** (the seam is graph-safe),
   with `--disable-cuda-graph` / `--attention-backend` as opt-in eager escapes.

**Verified on the H100 — all green:** 52 pytest; `optima compat` ALL SEAMS INTACT on 0.5.12;
block-slot op-correctness exact (CPU + CUDA, MHA/GQA/MQA). End-to-end on Qwen2.5-0.5B:
- silu gate (graphs on, fa3): faithful PASS (kl 0); broken silu **1.05× faster → FAIL** (kl 14.5),
  score 0.
- **Baseline strength:** graphs-off + triton = **2069 tok/s** vs graphs-on + fa3 = **13567 tok/s**
  (~6.5×). That gap is exactly why graphs-off is no longer the scoring default.
- **Decode-attention swap** (eager, `OPTIMA_ATTENTION_SEAM=1`): the seam extracts the *live*
  model's paged KV and routes decode to the miner kernel. Faithful kernel reproduces the model
  (PASS, kl ~6e-3 under a calibrated gate); broken decode (drops the `seq_lens` mask) caught ~20×
  (kl 0.126).

**Findings worth keeping:**
- **Attention has a higher intrinsic KL floor than elementwise ops:** *any* reference SDPA sits at
  **~6e-3 mean KL vs fa3's flash attention** (flash's online-softmax reduction rounds differently,
  and it compounds over layers) — stable across kernel precision (fp32 / bf16-SDPA) and backend
  (fa3 / torch_native). So the default 5e-3 gate (tuned for silu/rmsnorm) is too strict for
  attention → **per-slot KL thresholds needed.** The gather is provably correct (op-correctness
  exact; KL stable; broken is 20× worse).
- **3 real bugs only surfaced on GPU:** a CUDA device-mismatch in the attention mask
  (`torch.arange` on CPU), the circular-import seam noise, and the 0.5.12 loader crash. py_compile
  / CPU tests missed all three — run on hardware before trusting a kernel-path change.
- `disable_cuda_graph=True` does **not** disable 0.5.12's piecewise / torch.compile path.

**Next rungs (priority):**
1. **Paged-direct decode attention** — a graph-safe contract (miner consumes the page table +
   pool buffers directly, no variable-shape gather), so it runs under CUDA graphs and can actually
   compete on speed vs fa3 / FlashMLA. (The current gather MVP is correctness-only, eager-only.)
2. **Per-slot KL thresholds** (EvalConfig is one global threshold today).
3. Then the bigger surfaces from the design review: **MoE grouped-GEMM** and dense **FP8 / FP4
   GEMM** slots — where the real B200 headroom is.

**Design conclusions (from the review + SGLang-optimization research this session):**
- A "slot" should be a SGLang **swappable operator** (attention / MoE-experts / GEMM / sampler),
  typed tensor-in/out, **strictly upstream of the logprobs/sampler** — that line is what keeps the
  API-substitution attack structurally dead. Below it (silu/rmsnorm) is too small to move tok/s;
  at/above the sampler re-opens substitution.
- Most of "how SGLang hits ~10k tok/s" is **system work** (PD-disaggregation, large-scale EP,
  RadixAttention, CUDA graphs, vLLM-V1-style process arch) that **isn't a kernel a miner can
  submit** — the validator owns that as a maintained substrate; miners compete on the kernel slice
  *within* it. Kernel slots are most valuable at the **frontier** (new arch/dtype/hardware where
  sglang only has a slow fallback).
- The seam **is** the backend-swap mechanism: a pinned, *unmodified* sglang patched at runtime, so
  we never fork sglang and every validator runs the same package (consensus). The gitignored
  `sglang/` clone is a dev reference only.
