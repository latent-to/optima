"""A/B decode-throughput harness for sglang's TP all-reduce backend — the collective ceiling.

Measures SUSTAINED, STEADY-STATE decode tok/s for one all-reduce backend. We do NOT trust
warmup/transient numbers on this hardware: clocks ramp and can't be locked (±6-17% swings that
once faked a "win" — the split_k artifact). Instead, run K sequential rounds of a long fixed-length
decode under sustained load, discard the first WARMUP_ROUNDS while the GPU ramps to steady clocks,
and report the mean + spread of the STEADY rounds.

* A tight steady spread (intra-backend, the `spread%` field) is the signal the number is
  trustworthy — if it's wide, the box never settled and the comparison is moot.
* The cross-backend comparison is steady-mean vs steady-mean. `default` (the custom all-reduce
  `two_shot`) is bookended FIRST and LAST in the sweep so any whole-run drift is visible: a
  candidate only "wins" if it clears `default` by more than the default-to-default gap.

The swept dimension is the TP all-reduce implementation, via stock sglang flags (no source patch),
so one pinned package — consensus preserved. Goal: does any stock backend (NVLS / symmetric-memory
in-network reduce) beat `two_shot` at sustained decode, before we build a collective seam + kernel.

Env (sweep.sh orchestrates):
    ALLREDUCE_BACKEND  default|nccl|nccl_nvls|symm_mem|torch_symm_mem|mscclpp
    ENGINE_KWARGS_JSON path to a json of base engine kwargs (B200 V4 needs swa_full_tokens_ratio
                       + moe_runner_backend=flashinfer_mxfp4); the backend kwargs are applied on top.
    MODEL_PATH (deepseek-ai/DeepSeek-V4-Flash)  TP (4)  MEM_FRACTION (0.85)
    BATCH (128)  ROUND_TOKENS (1024)  ROUNDS (6)  WARMUP_ROUNDS (2)

A flag *requests* a backend; sglang's per-message should_*() decides which kernel runs. Confirm the
kernel that actually ran out-of-band with nsys where CUPTI is available (CC-mode boxes block it).
"""

from __future__ import annotations

import os
import time

import sglang as sgl

# backend -> the stock sglang Engine kwargs that select that TP all-reduce path.
BACKENDS: dict[str, dict] = {
    "default": {},  # custom all-reduce (the `two_shot` kernel) — baseline
    "nccl": {"disable_custom_all_reduce": True},  # drop custom AR -> NCCL ring / LL
    "nccl_nvls": {"disable_custom_all_reduce": True, "enable_nccl_nvls": True},  # NCCL NVSwitch in-network reduce
    "symm_mem": {"enable_symm_mem": True},  # pynccl symmetric-memory (NVLS) path
    "torch_symm_mem": {"enable_torch_symm_mem": True},  # PyTorch multimem all-reduce
    "mscclpp": {"enable_mscclpp": True},  # mscclpp small-message AR (falls back to NCCL)
}

_PROMPT = "Explain step by step how an out-of-order superscalar CPU executes instructions."


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


def main() -> None:
    backend = _env("ALLREDUCE_BACKEND", "default")
    if backend not in BACKENDS:
        print(f"RESULT backend={backend} : BAD_BACKEND (known: {','.join(BACKENDS)})", flush=True)
        return

    batch = int(_env("BATCH", "128"))
    round_tokens = int(_env("ROUND_TOKENS", "1024"))
    rounds = int(_env("ROUNDS", "6"))
    warmup_rounds = int(_env("WARMUP_ROUNDS", "2"))

    kw = dict(
        model_path=_env("MODEL_PATH", "deepseek-ai/DeepSeek-V4-Flash"),
        tp_size=int(_env("TP", "4")),
        trust_remote_code=True,
        mem_fraction_static=float(_env("MEM_FRACTION", "0.85")),
        moe_runner_backend=_env("MOE_BACKEND", "marlin"),
        chunked_prefill_size=4096,
        disable_flashinfer_autotune=True,
    )
    # Per-box base engine config merged over the defaults — e.g. B200 V4-Flash needs
    # swa_full_tokens_ratio + moe_runner_backend=flashinfer_mxfp4 to even init. Point
    # ENGINE_KWARGS_JSON at the same json the working run used; the swept all-reduce
    # backend is applied LAST so it always wins.
    ek_path = _env("ENGINE_KWARGS_JSON", "")
    if ek_path:
        import json

        with open(ek_path) as f:
            kw.update(json.load(f))
    kw.update(BACKENDS[backend])

    print(
        f"CONFIG backend={backend} applied={BACKENDS[backend]} | model={kw['model_path']} "
        f"tp={kw['tp_size']} moe={kw['moe_runner_backend']} batch={batch} round_tokens={round_tokens} "
        f"rounds={rounds} warmup={warmup_rounds}",
        flush=True,
    )

    try:
        engine = sgl.Engine(**kw)
    except Exception as ex:  # noqa: BLE001 - a backend may be unavailable on this box; record and move on
        print(f"RESULT backend={backend} : ENGINE_FAILED {type(ex).__name__}: {str(ex)[:160]}", flush=True)
        return

    # ignore_eos forces every sequence to exactly round_tokens tokens -> clean fixed-size rounds.
    sp = {"temperature": 0.0, "max_new_tokens": round_tokens, "ignore_eos": True}
    steady: list[float] = []
    for r in range(rounds):
        try:
            t0 = time.time()
            engine.generate([_PROMPT] * batch, sp)
            dt = time.time() - t0
            tps = batch * round_tokens / dt
            tag = "warmup" if r < warmup_rounds else "steady"
            if tag == "steady":
                steady.append(tps)
            print(f"ROUND backend={backend:<14} r={r} : {tps:8.1f} tok/s ({dt:5.1f}s) [{tag}]", flush=True)
        except Exception as ex:  # noqa: BLE001
            print(f"ROUND backend={backend:<14} r={r} : FAILED {type(ex).__name__}: {str(ex)[:120]}", flush=True)

    if steady:
        mean = sum(steady) / len(steady)
        lo, hi = min(steady), max(steady)
        spread = 100 * (hi - lo) / mean if mean else 0.0
        print(
            f"RESULT backend={backend:<14} : steady {mean:8.1f} tok/s  "
            f"(min {lo:.1f} max {hi:.1f} spread {spread:.1f}% n={len(steady)} | batch={batch} round_tok={round_tokens})",
            flush=True,
        )
    else:
        print(f"RESULT backend={backend:<14} : NO_STEADY_ROUNDS", flush=True)

    engine.shutdown()


if __name__ == "__main__":
    main()
