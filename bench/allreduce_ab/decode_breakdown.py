"""Decode-only kernel breakdown from an nsys sqlite — the B200 (or any-box) map.

Generalizes parse_allreduce_latency.py from "just comm" to the full picture: splits decode
(cudagraph-replayed: ``graphId`` set) from prefill (eager), then for the DECODE path prints
(1) a category rollup — comm / attention / moe_gemm / gemm / moe_glue / quant / norm_hc /
elementwise / other — and (2) the top-N individual kernels with their % of decode time and
category tag. The category rollup IS the lever map: it tells you, for THIS box and config,
how much of decode is communication vs the MoE GEMM vs attention vs glue, so you size targets
before building.

Every hard number we have is H200; run this on the B200 pod's nsys output to get the B200 map.
It consumes the same sqlite that ``sweep.sh NSYS=1`` produces (e.g. results/ar_01_default.sqlite),
or any nsys run exported with ``nsys export --type sqlite``.

Usage: python3 decode_breakdown.py RUN.sqlite [RUN2.sqlite ...] [--top N] [--prefill]

Caveats (name-heuristic, so read with judgment):
* ``deep_gemm`` serves BOTH the MLA/dense projections AND the MoE experts — it lands in the
  generic ``gemm`` bucket; don't read all of ``gemm`` as MoE.
* categories match on demangled-name substrings in priority order; ``other`` = unmatched.
"""

from __future__ import annotations

import sqlite3
import sys

# Priority-ordered: first matching category wins. Substrings matched against the lowercased name.
CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("comm", ("llreduce", "all_reduce", "allgather", "reducescatter", "nccl", "nvls",
              "multimem", "deep_ep", "::dispatch", "::combine", "sendrecv", "symm")),
    ("attention", ("flash_fwd", "_mla", "sparse_attn", "splitkv", "mla_combine", "get_mla",
                   "mla_metadata", "fmha", "paged_mqa", "attention")),
    ("moe_glue", ("expandinputrows", "finalizemoerouting", "doactivation", "swiglu",
                  "blockexpertprefixsum", "moe_align", "moe_sum", "permute", "topk", "routing")),
    ("moe_gemm", ("marlin", "gemmuniversal", "group_gemm", "grouped", "fused_moe", "trtllm")),
    ("gemm", ("deep_gemm", "cutlass", "nvjet", "cublas", "matmul", "_mma", "gemm")),
    ("quant", ("per_token_group_quant", "quant", "scaled_mm", "fp8_e4m3", "to_fp8")),
    ("norm_hc", ("rmsnorm", "layernorm", "_norm", "mhc", "hc_pre", "hc_post", "sinkhorn", "tilelang")),
    ("rope", ("rope", "rotary")),
    ("elementwise", ("elementwise", "vectorized", "reduce_kernel", "catarray", "direct_copy",
                     "clamp", "silu", "_add", "fill", "memset", "index_")),
]

_DECODE = "graphId IS NOT NULL AND graphId != 0"
_PREFILL = "graphId IS NULL OR graphId = 0"

_KERNELS = """
SELECT s.value AS name, COUNT(*) AS n, SUM(k.end - k.start) AS dur_ns,
       MIN(k.end - k.start) AS min_ns, MAX(k.end - k.start) AS max_ns
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON s.id = k.demangledName
WHERE {where}
GROUP BY s.value
"""


def categorize(name: str) -> str:
    low = name.lower()
    for cat, needles in CATEGORIES:
        if any(nd in low for nd in needles):
            return cat
    return "other"


def report(path: str, top_n: int, show_prefill: bool) -> None:
    con = sqlite3.connect(path)
    try:
        print(f"\n========== {path} ==========")
        for label, where in ((("decode", _DECODE),) + ((("prefill", _PREFILL),) if show_prefill else ())):
            rows = list(con.execute(_KERNELS.format(where=where)))
            total = sum(r[2] for r in rows) or 1
            print(f"\n--- {label.upper()} : {total / 1e6:.1f} ms GPU time, {len(rows)} distinct kernels ---")

            cat_ms: dict[str, float] = {}
            for name, _n, dur, _mn, _mx in rows:
                cat_ms[categorize(name)] = cat_ms.get(categorize(name), 0.0) + dur / 1e6
            print("  category rollup (the lever map):")
            for cat, ms in sorted(cat_ms.items(), key=lambda kv: -kv[1]):
                print(f"    {cat:<12} {ms:9.1f} ms  {100 * ms / (total / 1e6):5.1f}%")

            print(f"  top {top_n} kernels:")
            print("    %5s %8s %8s %8s %-11s %s" % ("%dec", "tot_ms", "avg_us", "max_us", "category", "kernel"))
            for name, n, dur, _mn, mx in sorted(rows, key=lambda r: -r[2])[:top_n]:
                print("    %5.1f %8.1f %8.1f %8.1f %-11s %s"
                      % (100 * dur / total, dur / 1e6, dur / 1e3 / n, mx / 1e3, categorize(name), name[:62]))
    finally:
        con.close()


def main() -> None:
    args = sys.argv[1:]
    if not args or "-h" in args or "--help" in args:
        print("usage: decode_breakdown.py RUN.sqlite [RUN2.sqlite ...] [--top N] [--prefill]")
        return
    top_n = 20
    if "--top" in args:
        i = args.index("--top")
        top_n = int(args[i + 1])
        del args[i:i + 2]
    show_prefill = "--prefill" in args
    paths = [a for a in args if a != "--prefill"]
    for path in paths:
        try:
            report(path, top_n, show_prefill)
        except Exception as ex:  # noqa: BLE001
            print(f"\n========== {path} ==========\nERROR: {type(ex).__name__}: {ex}")


if __name__ == "__main__":
    main()
