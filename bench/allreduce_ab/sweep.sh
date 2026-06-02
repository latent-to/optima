#!/usr/bin/env bash
# Sweep sglang's TP all-reduce backends and measure DECODE tok/s for each — the collective
# ceiling. Question answered: does any STOCK backend beat the default custom all-reduce
# (`two_shot`) at decode shapes, before we build a collective seam + custom kernel?
#
# Ordering-controlled: `default` is bookended FIRST and LAST. If the two `default` runs differ
# by more than the candidate-vs-default deltas, the box is noise-bound (clocks ramping) — lock
# clocks (`nvidia-smi -lgc`, if permitted) or re-run; do NOT trust a "win" smaller than the
# default-to-default spread. (This is the warmup-artifact discipline from the split_k saga.)
#
# Run INSIDE the sglang container. Bare invocation:
#     BACKENDS="default nccl nccl_nvls symm_mem torch_symm_mem mscclpp default" \
#     MODEL_PATH=deepseek-ai/DeepSeek-V4-Flash TP=4 MOE_BACKEND=marlin NSYS=1 \
#     bash bench/allreduce_ab/sweep.sh
# Docker wrapper: see README.md (H200 = lmsysorg/sglang:latest, B200 = :deepseek-v4-blackwell).
#
# With NSYS=1 each backend is profiled and parse_allreduce_latency.py prints the per-call
# all-reduce latency (decode-only) — the ground truth for which kernel actually ran and how
# close it is to the bandwidth floor.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BACKENDS="${BACKENDS:-default nccl nccl_nvls symm_mem torch_symm_mem mscclpp default}"
NSYS="${NSYS:-0}"
OUT="${OUT:-$HERE/results}"
mkdir -p "$OUT"

export MODEL_PATH="${MODEL_PATH:-deepseek-ai/DeepSeek-V4-Flash}"
export TP="${TP:-4}"
export MOE_BACKEND="${MOE_BACKEND:-marlin}"
export MEM_FRACTION="${MEM_FRACTION:-0.85}"
# sustained steady-state knobs (decode_bench runs ROUNDS rounds of BATCH x ROUND_TOKENS,
# discards WARMUP_ROUNDS while clocks ramp, reports the steady mean + spread).
export BATCH="${BATCH:-128}"
export ROUND_TOKENS="${ROUND_TOKENS:-1024}"
export ROUNDS="${ROUNDS:-6}"
export WARMUP_ROUNDS="${WARMUP_ROUNDS:-2}"

echo "== allreduce A/B (sustained) == model=$MODEL_PATH tp=$TP moe=$MOE_BACKEND batch=$BATCH round_tok=$ROUND_TOKENS rounds=$ROUNDS(warmup $WARMUP_ROUNDS) nsys=$NSYS"
echo "== order (default bookended first & last): $BACKENDS"

i=0
for backend in $BACKENDS; do
  i=$((i + 1))
  tag="$(printf '%02d_%s' "$i" "$backend")"
  echo "########## $tag ##########"
  if [ "$NSYS" = "1" ]; then
    rep="$OUT/ar_$tag"
    nsys profile --force-overwrite=true --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --cuda-graph-trace=node -o "$rep" \
      env ALLREDUCE_BACKEND="$backend" python3 "$HERE/decode_bench.py" 2>&1 \
      | grep -aE "CONFIG|ROUND|RESULT|ENGINE_FAILED"
    nsys export --type sqlite --force-overwrite true --output "$rep.sqlite" "$rep.nsys-rep" >/dev/null 2>&1 \
      && python3 "$HERE/parse_allreduce_latency.py" "$rep.sqlite" || echo "  (nsys export/parse skipped)"
  else
    env ALLREDUCE_BACKEND="$backend" python3 "$HERE/decode_bench.py" 2>&1 \
      | grep -aE "CONFIG|ROUND|RESULT|ENGINE_FAILED"
  fi
done

echo "ALLREDUCE_AB_DONE"
