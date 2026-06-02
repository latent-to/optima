# All-reduce A/B — the collective ceiling

**Question this answers:** in DeepSeek-V4-Flash decode, the TP all-reduce is ~**34% of decode
GPU time** (champion = FP4 marlin, TP=4 on H200) and it is **latency-bound, not bandwidth-bound**
(the decode `all_reduce_two_shot` kernel ran at avg ~333 µs vs a ~9 µs bandwidth floor — see the
2026-06-02 H200 nsys profile). Before we build a *collective seam* (a new `collective.all_reduce`
slot) and a custom low-latency reduce, we must know: **does any stock sglang all-reduce backend
already beat the default `two_shot` at decode shapes?** If yes, that backend is the ceiling to
beat (or a free config win). If no — and the default is already near the floor — the standalone
reduce is closed and only **compute-comm overlap** (the fenced escape-hatch tier) is left.

This harness measures that ceiling with **zero source patches** — every backend is a stock sglang
flag, so one pinned package, consensus preserved.

## Files
- `decode_bench.py` — measures SUSTAINED steady-state decode tok/s for ONE backend: K rounds of
  BATCH×ROUND_TOKENS decode, discard the warmup rounds (clocks ramp), report the steady mean +
  `spread%`. We don't trust transient/warmup numbers on unlockable-clock boxes — a tight steady
  spread is the trust signal. Backend via `ALLREDUCE_BACKEND`; per-box engine config (B200 V4 needs
  `swa_full_tokens_ratio` + `flashinfer_mxfp4`) via `ENGINE_KWARGS_JSON`.
- `sweep.sh` — runs the backend matrix, `default` bookended first & last (warmup-drift control).
  With `NSYS=1`, profiles each and prints per-call latency via the parser.
- `parse_allreduce_latency.py` — decode-only (graphId-split) per-call all-reduce latency from an
  nsys sqlite. `avg_us >> min_us` ⇒ latency-bound (the lever is latency/overlap, not bytes).
- `decode_breakdown.py` — decode-only **category rollup** + top-N kernels from an nsys sqlite (the
  full lever map: comm vs moe_gemm vs attention vs glue vs gemm …). Run it on the B200 default-backend
  capture to get the B200 map — every hard number we have is H200. Validated on the H200 captures:
  reproduces marlin decode = moe_gemm 41% / comm 35% / attention 5% / glue 0.6%, and flashinfer =
  moe_gemm 43% / comm 35% / glue 2.4% (marlin fuses the glue; flashinfer does not).

## Backend matrix (all stock sglang flags)

| `ALLREDUCE_BACKEND` | sglang Engine kwargs | exercises |
|---|---|---|
| `default` | *(none)* | custom all-reduce — the `two_shot` kernel (baseline) |
| `nccl` | `disable_custom_all_reduce=True` | NCCL ring / LL |
| `nccl_nvls` | `disable_custom_all_reduce, enable_nccl_nvls` | NCCL **NVSwitch in-network** reduce |
| `symm_mem` | `enable_symm_mem=True` | pynccl symmetric-memory (NVLS) path |
| `torch_symm_mem` | `enable_torch_symm_mem=True` | PyTorch **multimem** all-reduce |
| `mscclpp` | `enable_mscclpp=True` | mscclpp small-message AR (NCCL fallback) |

A flag *requests* a backend; sglang's per-message `should_*()` predicate decides which kernel
actually runs each call. **Always confirm with `NSYS=1`** (the parser shows the real kernel name).

## Run

Inside the sglang container (H200 = `lmsysorg/sglang:latest`, B200 = `lmsysorg/sglang:deepseek-v4-blackwell`):

```bash
# H200 champion (FP4 marlin, TP=4), with per-call latency:
MODEL_PATH=deepseek-ai/DeepSeek-V4-Flash TP=4 MOE_BACKEND=marlin NSYS=1 \
  bash bench/allreduce_ab/sweep.sh

# B200 (FP4 has native tensor cores -> MoE GEMM shrinks -> comm is an even bigger share):
MODEL_PATH=deepseek-ai/DeepSeek-V4-Flash TP=4 MOE_BACKEND=flashinfer_mxfp4 NSYS=1 \
  bash bench/allreduce_ab/sweep.sh
```

Docker (mount the repo, `--shm-size 32g` is mandatory for TP NCCL — omitting it gives
"NCCL unhandled system error"):

```bash
docker run --rm --gpus all --privileged --ipc=host --shm-size 32g --network host \
  -e CUDA_HOME=/usr/local/cuda -e TORCH_CUDA_ARCH_LIST=9.0 \
  -v "$PWD":/opt/optima -v /root/.cache:/root/.cache -v /root/models:/root/models \
  lmsysorg/sglang:latest \
  bash -lc 'cd /opt/optima && NSYS=1 MOE_BACKEND=marlin bash bench/allreduce_ab/sweep.sh'
```
(`TORCH_CUDA_ARCH_LIST`: 9.0 = H100/H200, 10.0 = B200.)

### Get the B200 map (we're blind there — all hard numbers are H200)
After a `NSYS=1` sweep, the `default`-backend capture is the B200 decode baseline:

```bash
python3 bench/allreduce_ab/decode_breakdown.py      bench/allreduce_ab/results/ar_01_default.sqlite
python3 bench/allreduce_ab/parse_allreduce_latency.py bench/allreduce_ab/results/ar_01_default.sqlite
```

The rollup tells you how B200 decode actually splits (comm / moe_gemm / attention / glue / …) so we
size every lever before building. On B200 native FP4 shrinks the MoE GEMM, so comm should be an even
larger share than the ~35% measured on H200.

## How to read it (the decision rule)

1. **Check the noise floor first.** Compare the two `default` runs (first vs last). The
   candidate-vs-default delta must exceed that spread to be real. If it doesn't, lock clocks or
   re-run — this is the warmup-artifact trap.
2. **A backend wins** if its decode tok/s clears `default` by more than the noise floor **and**
   (under `NSYS=1`) its per-call `avg_us` is below `default`'s. That backend is the ceiling.
   - If it's a *pure flag flip* with no fidelity cost, it's a config win — flip it, no seam needed.
   - If beating it needs a *novel* kernel, that's the `collective.all_reduce` seam target.
3. **No backend wins** and `default`'s `avg_us` is near `min_us` (the ~9 µs floor) ⇒ the standalone
   reduce is at the HW limit. Stop here on the standalone path; the remaining comm lever is
   **compute-comm overlap** (fuse the down-proj GEMM with reduce-scatter), which is the escape-hatch
   / framework tier per `docs/SLOT_CONTRACT.md`, not a core slot.

## Scope
Targets the **TP all-reduce** path (FP4 marlin/flashinfer, the H200 champion). The FP8 config's
comm is **DeepEP all-to-all** (`--moe-a2a-backend deepep --enable-dp-attention`), a different
collective — a separate study, not covered here.
