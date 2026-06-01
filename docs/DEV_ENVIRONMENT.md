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

Current lium pods run without `CAP_SYS_ADMIN`, so in-process
`unshare(CLONE_NEWNET)` no-egress isolation fails with `EPERM` there. Use
`--allow-unsafe-no-isolation` only for dev throughput replication. Production
validator scoring should run the eval worker in a privileged namespace-capable
container/VM, or launch the candidate side under Docker/OCI with `--network=none`
and the required GPU mounts.

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

## Session log

> Newest first. A running record of what each agent/session actually did, so the
> next one can resume cold. Candid and concrete — commands, numbers, gotchas.

### 2026-06-01 (later) — MoE block seam: MXFP4 win at forked-backend parity, through SlotSpec (Opus 4.8)

**Box:** 4× RTX PRO 6000 Blackwell (sm120, ~96 GB ea), venv `.venv-sglang-latest-cu130`
(torch 2.12.0+cu130, sglang 0.5.12.post1, flashinfer 0.6.12, triton_kernels). gpt-oss-120b
weights cached. Put the venv bin + `/usr/local/cuda/bin` on `PATH` so flashinfer's JIT finds
`ninja` + `nvcc` (else `FileNotFoundError: ninja`).

**Shipped (branch `feat/moe-block-seam-mxfp4-sm120`):**

1. **`FusedMoE.forward` BLOCK seam** (`dispatch.make_moe_dispatcher` + `integrations/sglang_moe.py`):
   routing (`topk_output`) is computed upstream, so the seam sits exactly at the expert-forward
   boundary — the MoE analogue of the `RadixAttention.forward` seam. Eager-only, non-EP, opt-in via
   `OPTIMA_MOE_SEAM=1`. The registry now carries the miner `prepare` at runtime, and a new SlotSpec
   `prepare_from_layer` hook maps the live layer → prepare args (the live gpt-oss MoE layer is
   **dequantized to bf16** by the triton fallback and carries biases — the 2-tensor contract
   couldn't express that). `OPTIMA_STRICT=1` surfaces kernel errors instead of silent fallback.
2. **`cosine` correctness mode** (`slots.Correctness` + `verify`): element-wise tolerance is
   meaningless at ~6–12 % fp4 per-element error; gate on cosine vs the fp32 reference. mxfp4 slot
   `min_cosine=0.97` (measured floor 0.985; a mis-ordered/broken kernel ≈ 0).
3. **Real autotuned MXFP4 fused-MoE bundle** (`examples/miner_moe_mxfp4_sm120`, stub overwritten,
   rebuild.json removed): prepare = de-interleave HF `[gate0,up0,…]` → CUTLASS `[up;gate]`, pad,
   pack MXFP4, interleave scales, carry biases; forward = MXFP8 act-quant + flashinfer
   `cutlass_fused_moe`.

**Numbers (gpt-oss-120b, TP=4, batch 32, eager, `gptoss_tp_bench` methodology):**
- Stock sglang on sm120 is **forced to the triton MoE fallback** — `flashinfer_cutlass`/`_trtllm`/
  `_mxfp4` all crash (`Mxfp4MoEMethod object has no attribute 'runner'`; cf. flashinfer #2577,
  sglang MoE "always resolves to Triton on SM120"). triton: **742 eager / 767 graphs**.
- seam → cutlass MXFP4, **untuned 874 → autotuned 912 median / 922 best**.
- hand-forked `flashinfer_mxfp4` (the experiment, patches sglang source): **926**.
- ⇒ the seam **matches the forked backend (~99 %) with NO fork**, +19 % over stock-realizable.
- Fidelity: cosine 0.985 vs fp32 / 0.999 vs dequant; live output coherent; strict mode confirms
  every layer ran the kernel (no fallback).

**The autotune gotcha (the entire 874→912 gap):** sglang's startup `_flashinfer_autotune()` only
profiles the *configured* MoE backend (triton here) — it never sees the injected cutlass call, so
it ran an untuned default tactic. Fix: tune once per problem shape under `autotune(True)` in the
kernel, then hit the process-global `AutoTuner` cache. Also: `prepare` OOMs at `mem_fraction 0.85`
(mxfp4 copies alongside the bf16 weights) — run ~0.65, `del` the padded scratch, `empty_cache()`.

**Running it (matters — the win is config-sensitive):** the lazy first-forward `prepare`
needs GPU headroom (it pads/quantizes dense bf16 experts while the model is resident). Two
supported configs, both eager:
- **eval default** `mem_fraction_static≈0.6` (what `EvalConfig` uses) — works as-is, 912–915 tok/s.
- **high mem** (`0.85`): set full-eager (`disable_cuda_graph` **and** `disable_piecewise_cuda_graph`
  — the piecewise-graph buffers otherwise eat the headroom) **and** `OPTIMA_MOE_FREE_DENSE=1`, which
  reclaims the dense bf16 experts after prepare (the kernel owns its MXFP4 copies). Verified: 920
  tok/s, no OOM. The production-clean fix (run at any mem) is load-time weight conversion — tracked.

**Codex review of PR #9 — findings + resolutions:**
1. *P1 live OOM during prepare* — was at `mem_fraction 0.85` **without** the memory settings (GPU
   ~full → the 1.08 GiB padded-bf16 transient can't allocate). Fixed: `OPTIMA_MOE_FREE_DENSE=1` +
   full-eager runs at 0.85 (920 tok/s); the eval's 0.6 works without freeing. Memory-lifecycle, not
   architecture (codex agreed).
2. *Docs overstated parity* — fair; phrasing now states the config (eager + mem headroom / free-dense)
   and that parity is 912–920 vs 926.
3. *Needs eager* — confirmed; `disable_piecewise_cuda_graph` is recommended (and required for
   free-dense safety: a fallback after freeing would hit empty weights, loudly). Documented.
4. *pytest missing in the pod venv* — true (uv venv, no pytest/pip); `py_compile`/`compat`/CUDA
   `verify` all pass. Run the suite with `uv pip install pytest` (or a minimal shim): 63 pass.
5. *rebuild.py remains an arbitrary-bundle-Python escape hatch* — out of scope here (removed it from
   THIS bundle); tracked as a separate hardening item (drop `bundle_python` / require vetted
   repo-local + content-pinned patchers).

**Next:** B200/sm100 (where sglang's FP4 MoE genuinely works + is heavily tuned — the real arena);
load-time weight conversion (run at any mem_fraction); a CUDA-graph-capturable seam.

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
  sglang only has a slow fallback — e.g. the sm_120 MoE work in `experiments/`).
- The seam **is** the backend-swap mechanism: a pinned, *unmodified* sglang patched at runtime, so
  we never fork sglang and every validator runs the same package (consensus). The gitignored
  `sglang/` clone is a dev reference only.
