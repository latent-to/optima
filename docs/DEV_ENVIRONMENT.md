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
| `brave-orbit-7c` | **4× RTX PRO 6000 Blackwell** (4×96 GB, GDDR7, no NVLink) | 13.0 | `154.54.100.130` | the **dev box** — TP / PD-disagg / EP / bigger models / Blackwell (FA4, nvfp4) |
| `golden-lion-b6` | H100 (80 GB HBM) | 13.0 | `216.81.245.218:40309` | where the harness was validated |

### Driving a pod programmatically

All from the `sn120` env (`lium` on PATH there):

```bash
lium ps                                   # list pods + names + IPs
lium exec brave-orbit-7c "nvidia-smi -L"  # run a command non-interactively
lium ssh  brave-orbit-7c                  # interactive shell
lium rsync brave-orbit-7c ./optima        # push a directory (use for the harness)
lium scp  brave-orbit-7c ./file /root/    # copy a single file
lium logs brave-orbit-7c                  # stream logs
lium rm   brave-orbit-7c                  # TERMINATE (stops billing)
```

The H100 also answers direct ssh: `ssh root@216.81.245.218 -p 40309 -i ~/.ssh/id_ed25519`.

> Billing: the Blackwell box is ~$4.64/h, the H100 ~$1.48/h. **Tear down idle pods
> with `lium rm <name>`** — they bill while RUNNING.

## Bootstrapping the harness on a fresh pod

The validated recipe (from the H100; adjust `TORCH_CUDA_ARCH_LIST` per GPU):

```bash
# on your machine: push the harness (NOT the sglang clone)
lium rsync brave-orbit-7c ~/Downloads/github/optima/optima
lium rsync brave-orbit-7c ~/Downloads/github/optima/examples

# on the pod (lium ssh, or wrap each in `lium exec brave-orbit-7c "..."`):
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
export TORCH_CUDA_ARCH_LIST=12.0                       # 12.0 = RTX PRO 6000 Blackwell (sm_120)
                                                       # 9.0 = H100, 10.0 = B200
.venv/bin/python -m optima.cli verify   examples/miner_silu_triton --device cuda
.venv/bin/python -m optima.cli evaluate examples/miner_silu_triton --model Qwen/Qwen2.5-0.5B-Instruct --no-deterministic
```

### Blackwell (sm_120) caveats to verify on first run

- sglang / sgl-kernel / flashinfer Blackwell (sm_120) support is **newer** than
  Hopper — expect possible build/runtime friction; `uv` may resolve a
  Blackwell-capable sgl-kernel/torch. If `pip install sglang` pulls a Hopper-only
  build, you may need a CUDA-13 / Blackwell wheel or a source build.
- GPT-OSS-120B on the Blackwell box is an active native-MXFP4 target under the
  pinned stack (`sglang==0.5.9`, `torch==2.9.1+cu128`, `flashinfer==0.6.3`).
  Stock behavior: `moe_runner_backend=auto` selects `flashinfer_mxfp4` but fails
  weight prep with `assert M % 128 == 0`; forcing `triton_kernel` hits a Triton
  `ptxas` failure for `sm_120a`; forcing plain `triton` works at TP=4 but OOMs at
  TP=2 because it expands MXFP4 weights.
  - Dev-pod result: the `flashinfer_mxfp4` path can be made correct on `sm_120a`.
    The necessary fixes were: pad GPT-OSS TP shards to FlashInfer's 256-wide
    shape, use plain packed FP4 weight bytes (no row shuffle), run FlashInfer's
    `nvfp4_block_scale_interleave` on the MXFP4 block-scale tensors, pass
    `swizzled_input_sf=False` for the activation scales, and disable PDL for the
    CUTLASS MoE call. Deterministic component probes moved from `cos ~0.86` to
    `cos ~0.9999`; TP=4 GPT-OSS output became coherent.
  - Current measured smoke on the 4× RTX PRO 6000 Blackwell pod, one prompt,
    64 generated tokens, warmup=1/timed=2: patched `flashinfer_mxfp4` median
    `46.0 tok/s`; plain Triton fallback median `38.9 tok/s`. Treat this as a
    promising dev result, not an upstreamed production baseline.
  - Latest-stack replication (`torch==2.12.0+cu130`,
    `sglang==0.5.12.post1`, `flashinfer-python==0.6.12`) also works after a
    smaller SGLang-only patch plus rebuilding `sglang-kernel` `common_ops` for
    Torch 2.12 / `sm_120a`. The key extra fix was TP=4 GPT-OSS MXFP4 loader
    padding: ranks receive 736 intermediate values from 32-value block-ceil
    checkpoint slicing even though SGLang's partition size is 720. Batch-32
    GPT-OSS-120B TP=4 decode, max-new=64, warmup=1/timed=3, piecewise CUDA graph
    disabled: the standardized runner passed with patched `flashinfer_mxfp4`
    median `915.6 tok/s`; plain `triton` median `742.6 tok/s`; speedup `1.23x`.
    A previous manual run measured `926.7 tok/s` versus `741.6 tok/s`.
  - Stronger best-stock check: the no-graph Triton number above is not the
    headline baseline. With CUDA graph enabled, explicit Triton attention,
    radix cache enabled, and custom all-reduce tested, the best stock SGLang
    batch-32 result so far is `830.7 tok/s` (`triton` MoE + `triton` attention +
    CUDA graph + radix). The patched `flashinfer_mxfp4` path under the symmetric
    CUDA-graph/radix setup measured `1062.4 tok/s` with custom all-reduce enabled,
    a `1.28x` speedup over the stronger stock baseline. Startup was ~126-129s for
    these CUDA-graph runs and is reported separately from timed decode throughput.
  - Long-context sanity: with prompts padded to ~2k tokens and forced 1024-token
    decode at batch 4, the same best-stock config measured `321.6 tok/s` while
    patched `flashinfer_mxfp4` measured `402.1 tok/s` (`1.25x`). Both runs found
    the expected fixed arithmetic answers in all timed iterations.
  - SGLang-packaged CUDA 13 envs are also viable for dependency hygiene:
    `uv --torch-backend cu130 --prerelease=allow sglang==0.5.12.post1`
    resolves to `torch==2.11.0+cu130`, `flashinfer-python==0.6.11.post1`,
    `sglang-kernel==0.4.2.post2`, and `xgrammar==0.2.0`. Loader smoke and the
    synthetic SwiGLU probe pass after the candidate patch, but the full
    `flashinfer_mxfp4` model path currently aborts inside FlashInfer/CUTE
    RMSNorm with `Expected an MLIR object`. That is the next packaged-stack
    blocker, separate from the MoE layout fix.
  - Plain Triton config tuning is not enough: a small fused-MoE config sweep on
    the GPT-OSS TP=4 shape found only ~1% isolated-kernel movement, and copied
    B200/RTX configs made the full TP=4 smoke slower than the no-config baseline.
- For multi-GPU work (the reason for this box): `--tp-size 4` etc. Throughput will
  be PCIe-comms-bound (no NVLink) — that's expected; the point is exploring the
  multi-GPU optimization surface (TP/PD/EP), not peak throughput.

## Clean-signal eval settings

When measuring kernel fidelity, run with `enable_deterministic_inference` so the
nondeterminism noise floor → ~0, and calibrate the KL threshold to a measured
stock-vs-stock floor (see `README.md` "Calibration findings"). Measure the floor
with `optima/.../noise_floor`-style stock-vs-stock runs before trusting a KL number.
