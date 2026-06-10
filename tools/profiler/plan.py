#!/usr/bin/env python3
"""Generate a bounded next-capture plan from a profiler dataset.

This is the antidote to profiling whack-a-mole. A clean profiling campaign has
three bounded products:
  * a serving truth pass (e2e + torch trace);
  * a timeline pass (nsys exports);
  * a counters pass (ncu target rows for categories that are still unknown).

``build.py`` tells us what is unknown. This script converts that into concrete
pod-script rows and backend A/Bs, so moving from Qwen to Nemotron/Minimax is a
repeatable checklist instead of hand-editing profiler commands from scratch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import findings as findings_mod  # noqa: E402


DEFAULT_DECODE_BENCH = "--batch-size 32 --input-len 8 --output-len 20 --disable-cuda-graph"
BS1_DECODE_BENCH = "--batch-size 1 --input-len 8 --output-len 20 --disable-cuda-graph"
LONGCTX_BENCH = "--batch-size 8 --input-len 65536 --output-len 8 --disable-cuda-graph"
PREFILL_BENCH = "--batch-size 2 --input-len 2048 --output-len 1 --disable-cuda-graph"


NCU_TARGETS = {
    "fp4_moe_gemm": {
        "label": "fp4gemm_b32",
        "section": "GEMM",
        "regex": r"bmm_.*E2m1",
        "skip": 250,
        "count": 8,
        "bench": DEFAULT_DECODE_BENCH,
        "reason": "decode FP4 MoE GEMM floor/shape check",
        "also": [
            {
                "label": "fp4gemm_b1",
                "section": "GEMM",
                "regex": r"bmm_.*E2m1",
                "skip": 250,
                "count": 8,
                "bench": BS1_DECODE_BENCH,
                "reason": "bs1 control for phantom occupancy",
            },
            {
                "label": "prefill_gemm",
                "section": "DEEP",
                "regex": r"bmm_.*E2m1",
                "skip": 120,
                "count": 10,
                "bench": PREFILL_BENCH,
                "reason": "prefill-vs-decode regime separation",
            },
        ],
    },
    "dense_gemm": {
        "label": "nvjet",
        "section": "GEMM",
        "regex": r"nvjet_sm",
        "skip": 250,
        "count": 10,
        "bench": DEFAULT_DECODE_BENCH,
        "reason": "projection GEMM bound-type and splitK parent shapes",
    },
    "splitk_reduce": {
        "label": "splitk",
        "section": "GEMM",
        "regex": r"splitKreduce",
        "skip": 250,
        "count": 8,
        "bench": DEFAULT_DECODE_BENCH,
        "reason": "associate splitK epilogue/reduction with parent GEMM",
    },
    "attention": {
        "label": "attn_16k",
        "section": "DEEP",
        "regex": r"fmha",
        "skip": 160,
        "count": 8,
        "bench": "--batch-size 32 --input-len 16384 --output-len 8 --disable-cuda-graph",
        "reason": "decode attention bound-type at normal context",
        "also": [
            {
                "label": "attn_64k",
                "section": "DEEP",
                "regex": r"fmha",
                "skip": 80,
                "count": 6,
                "bench": LONGCTX_BENCH,
                "reason": "long-context attention bound-type",
            }
        ],
    },
    "gdn_scan": {
        "label": "gdnscan",
        "section": "DEEP",
        "regex": r"gdn",
        "skip": 250,
        "count": 8,
        "bench": DEFAULT_DECODE_BENCH,
        "reason": "linear-attention recurrence bound-type",
    },
    "gdn_conv": {
        "label": "conv",
        "section": "STALL",
        "regex": r"causal_conv1d",
        "skip": 250,
        "count": 6,
        "bench": DEFAULT_DECODE_BENCH,
        "reason": "GDN state-update conv launch cost",
    },
    "fused_qkvzba": {
        "label": "qkvzba",
        "section": "STALL",
        "regex": r"qkvzba",
        "skip": 250,
        "count": 6,
        "bench": DEFAULT_DECODE_BENCH,
        "reason": "GDN projection split/reshape glue",
    },
    "moe_finalize": {
        "label": "finalize",
        "section": "GEMM",
        "regex": r"finalizeKernel",
        "skip": 250,
        "count": 6,
        "bench": DEFAULT_DECODE_BENCH,
        "reason": "MoE finalize fusion target",
    },
    "rmsnorm": {
        "label": "norm",
        "section": "STALL",
        "regex": r"norm",
        "skip": 250,
        "count": 6,
        "bench": DEFAULT_DECODE_BENCH,
        "reason": "norm launch/fusion target",
    },
    "act_mul": {
        "label": "act",
        "section": "STALL",
        "regex": r"act_and_mul",
        "skip": 250,
        "count": 6,
        "bench": DEFAULT_DECODE_BENCH,
        "reason": "activation multiply launch/fusion target",
    },
    "nvfp4_quant": {
        "label": "nvfp4_quant",
        "section": "STALL",
        "regex": r"fp4",
        "skip": 250,
        "count": 6,
        "bench": DEFAULT_DECODE_BENCH,
        "reason": "quant/scale-interleave glue",
    },
    "elementwise": {
        "label": "elementwise",
        "section": "STALL",
        "regex": r"elementwise",
        "skip": 250,
        "count": 8,
        "bench": DEFAULT_DECODE_BENCH,
        "reason": "generic PyTorch elementwise bucket; pair with NVTX callsites",
    },
}


COMM_NOTE = (
    "NCU supports multi-process/multi-GPU profiling, but communication kernels need "
    "the right mode: target all ranks/processes, use a communicator/lockstep launch "
    "when kernels must run concurrently, and prefer NVTX/range/application replay over "
    "naive single-rank kernel replay. Pair the counters with e2e A/B and nsys timeline."
)


def _load_dataset(path: Path) -> dict:
    data = json.loads(path.read_text())
    if "dataset" in data and isinstance(data["dataset"], dict):
        data = data["dataset"]
    if "findings" not in data or not data["findings"]:
        data["findings"] = findings_mod.derive(data)
    return data


def _canonical_categories(dataset: dict) -> list[dict]:
    return dataset.get("findings", {}).get("decode_canonical", {}).get("categories", [])


def _ncu_labels(dataset: dict) -> set[str]:
    return {c.get("label") for c in dataset.get("ncu", []) if c.get("label")}


def _server_args(dataset: dict, key: str) -> set[str]:
    rows = dataset.get("health", {}).get("sglang", {}).get("server_args", {}).get(key, [])
    return {str(r.get("value")) for r in rows}


def _gdn_needs_ab(dataset: dict) -> bool:
    for row in dataset.get("health", {}).get("sglang", {}).get("gdn_dispatchers", []):
        if row.get("decode_kernel") == "FlashInferGDNKernel" and str(row.get("packed_decode")) == "False":
            return True
    return False


def _rows_for_category(cat: str) -> list[dict]:
    spec = NCU_TARGETS.get(cat)
    if not spec:
        return []
    out = [{k: v for k, v in spec.items() if k != "also"}]
    out.extend(spec.get("also", []))
    return out


def plan(dataset: dict, min_pct: float = 0.8, include_known_wins: bool = True) -> dict:
    labels_done = _ncu_labels(dataset)
    ncu_rows: list[dict] = []
    seen: set[str] = set()
    comm_actions = []

    for cat in _canonical_categories(dataset):
        name = cat.get("cat")
        pct = float(cat.get("pct") or 0)
        if pct < min_pct:
            continue
        if name == "all_reduce":
            comm_actions.append({
                "title": f"Repeat all-reduce e2e/nsys A/B ({pct:.1f}% decode)",
                "reason": COMM_NOTE,
                "flags": [
                    "baseline: default/custom all-reduce",
                    "ablation: --disable-custom-all-reduce",
                    "ncu comm pass: --target-processes all plus --communicator shmem/tcp and --lockstep-kernel-launch for mandatory-concurrent kernels",
                    "topology-gated: --enable-symm-mem / --enable-nccl-nvls only if supported",
                ],
            })
            continue
        if cat.get("ncu"):
            continue
        if cat.get("winnable") is False and name not in ("dense_gemm", "fp4_moe_gemm"):
            continue
        if cat.get("winnable") is False and cat.get("bound_type") in ("memory", "compute") and name != "dense_gemm":
            continue
        if cat.get("winnable") is True and not include_known_wins:
            continue
        for row in _rows_for_category(name):
            if row["label"] in labels_done or row["label"] in seen:
                continue
            seen.add(row["label"])
            ncu_rows.append({**row, "category": name, "decode_pct": round(pct, 2)})

    backend_abs = []
    if _gdn_needs_ab(dataset):
        backend_abs.append({
            "name": "gdn_decode_backend",
            "reason": "Captured runs used FlashInferGDNKernel packed_decode=False; SGLang also has Triton packed decode.",
            "variants": [
                {"tag": "gdn_flashinfer", "flags": "--linear-attn-decode-backend flashinfer"},
                {"tag": "gdn_triton", "flags": "--linear-attn-decode-backend triton"},
            ],
        })

    if _server_args(dataset, "linear_attn_decode_backend") == {"None"}:
        backend_abs.append({
            "name": "linear_attn_decode_explicitness",
            "reason": "Decode backend was implicit. Pin it so future reports are comparable.",
            "variants": [
                {"tag": "linear_decode_pinned", "flags": "--linear-attn-decode-backend <chosen-backend>"},
            ],
        })

    unknown_pct = dataset.get("findings", {}).get("amdahl", {}).get("unknown_pct")
    stop = []
    if unknown_pct is not None:
        if unknown_pct <= 5 and not backend_abs:
            stop.append("Unknown decode surface is <=5% and no backend A/B is open: stop profiling and optimize.")
        else:
            stop.append(f"Unknown decode surface is {unknown_pct}%: close only the listed rows, then stop.")
    stop.append("Do not chase categories below the noise floor unless a patch moves e2e.")

    return {
        "ncu_rows": ncu_rows,
        "backend_abs": backend_abs,
        "comm_actions": comm_actions,
        "stop_conditions": stop,
    }


def render(plan_obj: dict) -> str:
    lines = ["# Capture Plan", ""]
    lines.append("## Backend A/Bs")
    if plan_obj["backend_abs"]:
        for ab in plan_obj["backend_abs"]:
            lines.append(f"- {ab['name']}: {ab['reason']}")
            for v in ab["variants"]:
                lines.append(f"  - {v['tag']}: `{v['flags']}`")
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## NCU Target Rows")
    lines.append("Pipe format: `label|section|regex|skip|count|bench args`")
    if plan_obj["ncu_rows"]:
        for r in plan_obj["ncu_rows"]:
            lines.append(
                f"{r['label']}|{r['section']}|{r['regex']}|{r['skip']}|{r['count']}|{r['bench']}"
                f"  # {r['category']} {r['decode_pct']}%: {r['reason']}"
            )
    else:
        lines.append("- none")
    lines.append("")

    lines.append("## Communication")
    if plan_obj["comm_actions"]:
        for c in plan_obj["comm_actions"]:
            lines.append(f"- {c['title']}: {c['reason']}")
            for f in c["flags"]:
                lines.append(f"  - {f}")
    else:
        lines.append("- no comm-specific action")
    lines.append("")

    lines.append("## Stop Conditions")
    for s in plan_obj["stop_conditions"]:
        lines.append(f"- {s}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset_json", type=Path, help="dataset.json from build.py")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--min-pct", type=float, default=0.8)
    ap.add_argument("--skip-known-wins", action="store_true")
    args = ap.parse_args()
    dataset = _load_dataset(args.dataset_json)
    text = render(plan(dataset, min_pct=args.min_pct, include_known_wins=not args.skip_known_wins))
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
