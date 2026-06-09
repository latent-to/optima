#!/usr/bin/env python3
"""Ingest a directory of GPU-profiling artifacts into one normalized dataset.

This is the front half of the profiler pipeline: it knows how to read every
artifact type we produce on a profiling box and flatten them into a single
JSON-serialisable dict. ``findings.py`` turns that dict into verdicts; the
``report.py`` renders it as a self-contained HTML dashboard.

Design constraints (learned the hard way):
  * **stdlib only** — the Mac checkout has no pandas and (usually) no NVIDIA
    CLIs, so ``.ncu-rep`` / ``.nsys-rep`` binaries are opaque. We read only
    their *text exports* (``_details.txt`` / ``_raw.csv`` / ``_kernsum.txt``).
  * **be honest about junk** — some ncu captures come back all-NaN (cluster /
    Cluster-Launch-Control kernels that ncu kernel-replay can't count) and some
    ``_summary.txt`` files are ``==ERROR==`` (wrong --page). We detect and flag
    those rather than silently averaging garbage.
  * **model-agnostic** — the category regexes are a config (``KERNEL_CATS``),
    overridable, so this works on the next model's profiles too, not just
    Qwen3.5.

CLI:  python3 ingest.py <datadir> [-o dataset.json]
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Kernel taxonomy. Order matters: first match wins (most-specific first).
# Override by passing a different mapping to ingest(..., kernel_cats=...).
# --------------------------------------------------------------------------- #
KERNEL_CATS: dict[str, re.Pattern] = {
    "fp4_moe_gemm": re.compile(r"^bmm_(?:E2m1|Bfloat16_E2m1)", re.I),
    "dense_gemm": re.compile(r"nvjet|cutlass.*gemm|gemm_", re.I),
    "splitk_reduce": re.compile(r"splitKreduce", re.I),
    "moe_finalize": re.compile(r"finalizeKernel", re.I),
    "moe_routing": re.compile(r"routingIndices|routingInit|routingCustom|moe.*topk", re.I),
    "all_reduce": re.compile(r"allreduce_fusion|all_reduce|AllReduce|ncclDevKernel_AllReduce|one_shot|lamport", re.I),
    "all_gather": re.compile(r"AllGather|_all_gather|all_gather", re.I),
    "delay_stream": re.compile(r"delayStreamKernel", re.I),
    "attention": re.compile(r"fmha|attention|mha|paged|flash_fwd|trtllm.*mha", re.I),
    "gdn_scan": re.compile(r"gdn_wide_vec|gated_delta|fused_recurrent|chunk_gated|chunk_scan", re.I),
    "gdn_conv": re.compile(r"causal_conv1d", re.I),
    "fused_qkvzba": re.compile(r"fused_qkvzba|qkvzba", re.I),
    "act_mul": re.compile(r"act_and_mul|silu|swiglu|sigmoid_gate", re.I),
    "rmsnorm": re.compile(r"rmsnorm|RMSNorm|layer_norm|LayerNorm|norm_fwd", re.I),
    "nvfp4_quant": re.compile(r"nvfp4_quantize|NVFP4Quantize|quantize.*fp4|block_scale_interleave|scale.*interleave", re.I),
    "token_gather": re.compile(r"vectorized_gather|gather_kernel|index_select|IndexKernel|gather", re.I),
    "kv_rope": re.compile(r"mrope|rope|set_kv_buffer|fp8_set_kv|reshape_and_cache", re.I),
    "sampling": re.compile(r"argmax|sample|softmax|penalt|resolve_future_token|logits", re.I),
    "copy_memset": re.compile(r"Memcpy|Memset|memcpy|memset|\bcopy_|elementwise_copy", re.I),
    "elementwise": re.compile(r"elementwise|triton_poi|FillFunctor|CUDAFunctor|float8_copy|add_kernel|mul_kernel", re.I),
}

# Human-readable labels for categories (UI + reports).
DISPLAY: dict[str, str] = {
    "fp4_moe_gemm": "FP4 MoE GEMM (bmm_*E2m1)",
    "dense_gemm": "dense/projection GEMM (nvjet)",
    "splitk_reduce": "splitKreduce GEMM epilogue",
    "moe_finalize": "MoE finalize",
    "moe_routing": "MoE routing",
    "all_reduce": "all-reduce (collective)",
    "all_gather": "all-gather (collective)",
    "delay_stream": "delayStreamKernel",
    "attention": "attention",
    "gdn_scan": "GDN scan (recurrence)",
    "gdn_conv": "GDN causal conv",
    "fused_qkvzba": "GDN qkvzba split/reshape",
    "act_mul": "act_and_mul / SiLU",
    "rmsnorm": "norm (rms/layer)",
    "nvfp4_quant": "NVFP4 quant / scale-interleave",
    "token_gather": "token gather / index",
    "kv_rope": "RoPE / KV write",
    "sampling": "sampling / logits",
    "copy_memset": "copy / memset",
    "elementwise": "elementwise misc",
    "other": "other / uncategorised",
}


def categorize(name: str, kernel_cats: dict[str, re.Pattern] = KERNEL_CATS) -> str:
    for cat, rx in kernel_cats.items():
        if rx.search(name):
            return cat
    return "other"


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _num(tok: str) -> float | None:
    """Parse one ncu metric token; '(!) nan' / 'no data' / 'n/a' -> None."""
    tok = tok.strip()
    if not tok or tok.lower() in ("nan", "no data", "n/a", "(!) nan", "<null>"):
        return None
    m = re.findall(r"[-+]?\d[\d,]*\.?\d*", tok)
    if not m:
        return None
    try:
        return float(m[-1].replace(",", ""))
    except ValueError:
        return None


def _pct(num: float, den: float) -> float:
    return 0.0 if den <= 0 else 100.0 * num / den


# --------------------------------------------------------------------------- #
# 1. torch profiler Chrome traces (*.trace.json.gz) — the clean DECODE source
# --------------------------------------------------------------------------- #
def _trace_label(annotations: collections.Counter, path: Path) -> str:
    names = set(annotations)
    if any("DRAFT_EXTEND" in n or "TARGET_VERIFY" in n for n in names):
        return "mtp_on"
    if any("DECODE" in n for n in names):
        return "mtp_off"
    return path.stem


def _trace_cache_path(path: Path) -> Path:
    st = path.stat()
    cdir = path.parent / ".profiler_cache"
    return cdir / f"{path.name}-{st.st_size}-{int(st.st_mtime)}.json"


def parse_torch_trace(path: Path, kernel_cats=KERNEL_CATS, use_cache: bool = True) -> dict:
    """Bucket GPU kernel time by category for one torch trace.

    Returns a per-trace summary. Kernel ``dur`` in Chrome traces is microseconds.
    Torch traces are huge (~1 GB JSON each); the summary is tiny, so we cache it
    in ``<datadir>/.profiler_cache/`` keyed by (size, mtime) — re-runs are instant
    while iterating on findings/report. Best-effort: cache errors are ignored.
    """
    if use_cache:
        cp = _trace_cache_path(path)
        if cp.exists():
            try:
                return json.loads(cp.read_text())
            except (OSError, ValueError):
                pass
    summary = _parse_torch_trace_uncached(path, kernel_cats)
    if use_cache:
        try:
            cp = _trace_cache_path(path)
            cp.parent.mkdir(exist_ok=True)
            cp.write_text(json.dumps(summary))
        except OSError:
            pass
    return summary


def _parse_torch_trace_uncached(path: Path, kernel_cats=KERNEL_CATS) -> dict:
    with gzip.open(path, "rt", errors="replace") as fh:
        data = json.load(fh)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data

    anns: collections.Counter = collections.Counter()
    cats: dict[str, list] = collections.defaultdict(lambda: [0.0, 0])   # cat -> [us, count]
    per_kernel: dict[str, list] = collections.defaultdict(lambda: [0.0, 0])
    total_us = 0.0
    launches = 0
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat_field = str(e.get("cat", "")).lower()
        if cat_field in ("user_annotation", "gpu_user_annotation"):
            anns[str(e.get("name", "?"))] += 1
            continue
        if cat_field not in ("kernel", "gpu_op", "gpu_memcpy", "gpu_memset"):
            continue
        dur = float(e.get("dur") or 0.0)
        name = str(e.get("name", "?"))
        c = categorize(name, kernel_cats)
        total_us += dur
        launches += 1
        cats[c][0] += dur
        cats[c][1] += 1
        per_kernel[name][0] += dur
        per_kernel[name][1] += 1

    rank = (re.search(r"TP-?(\d+)", path.name) or [None, "?"])[1]
    cat_rows = {
        c: {"us": v[0], "pct": _pct(v[0], total_us), "count": v[1]}
        for c, v in cats.items()
    }
    top = sorted(
        ({"name": n[:160], "cat": categorize(n, kernel_cats), "us": v[0],
          "pct": _pct(v[0], total_us), "count": v[1]} for n, v in per_kernel.items()),
        key=lambda r: -r["us"],
    )[:60]
    return {
        "file": path.name,
        "label": _trace_label(anns, path),
        "rank": f"TP{rank}",
        "total_us": total_us,
        "launches": launches,
        "cats": cat_rows,
        "top_kernels": top,
    }


# --------------------------------------------------------------------------- #
# 2. ncu text exports (_details.txt primary, _raw.csv for waves/registers)
# --------------------------------------------------------------------------- #
# header line ends with ", Context N, Stream N, Device N, CC X.Y"
_NCU_HDR = re.compile(r"^\s*(\S.*?)\s+\([\d,\s]+\)x\([\d,\s]+\),\s*Context\s+\d+,\s*Stream\s+\d+", re.I)
_NCU_METRICS = {
    "comp": "Compute (SM) Throughput",
    "mem": "Memory Throughput",
    "dram": "DRAM Throughput",
    "dur_us": "Duration",
    "occ": "Achieved Occupancy",
    "warps": "Achieved Active Warps Per SM",
    "l2_hit": "L2 Hit Rate",
    "l1_hit": "L1/TEX Hit Rate",
}


def _ncu_metric_line(line: str, label: str) -> float | None:
    """A metric row is '<Label> <unit> <value>'. Match label as a prefix of the
    stripped line so 'Memory Throughput' doesn't also catch 'L2 ... Throughput'."""
    s = line.strip()
    if not s.startswith(label):
        return None
    rest = s[len(label):]
    return _num(rest.split()[-1]) if rest.split() else None


_CLUSTER_NAME = re.compile(r"ClusterKernel|cluster_launch|_cluster_", re.I)


def parse_ncu_details(path: Path, kernel_cats=KERNEL_CATS) -> list[dict]:
    """One row per profiled kernel.

    Flags two kinds of untrustworthy capture so the findings layer never
    manufactures a phantom win from them:
      * ``valid=False`` — every metric came back NaN (ncu kernel-replay could
        not count it at all).
      * ``clc=True`` / ``cluster=True`` — the kernel uses thread-block clusters
        / Cluster-Launch-Control, which DEPRESSES the launched cluster/block/
        warp counts, so occupancy & throughput read artificially low (a fake
        "latency-bound, just fuse it" signal). These are vendor cluster kernels.
    """
    rows: list[dict] = []
    cur: dict | None = None

    def _flush():
        if cur is not None:
            cur["valid"] = any(cur.get(k) is not None for k in ("comp", "mem", "dram", "occ"))
            cur["cat"] = categorize(cur["kernel"], kernel_cats)
            cur.setdefault("clc", False)
            # `cluster` is STRUCTURAL (by name): a vendor cluster-dispatch kernel you
            # can't fuse (e.g. routingIndicesClusterKernel). `clc` is the CLC *warning*:
            # it corrupts occupancy/launch counts but NOT throughput — an FP4 cutlass
            # GEMM (2cta clusters) has clc=True yet its 80% DRAM reading is valid.
            # Keep them separate so we don't exclude a real memory-bound floor.
            cur["cluster"] = bool(_CLUSTER_NAME.search(cur["kernel"]))
            rows.append(cur)

    for line in path.read_text(errors="replace").splitlines():
        h = _NCU_HDR.match(line)
        if h:
            _flush()
            cur = {"kernel": h.group(1).strip()[:160]}
            continue
        if cur is None:
            continue
        if "Cluster Launch Control" in line or "CLC" in line and "feature enabled" in line:
            cur["clc"] = True
            continue
        for key, label in _NCU_METRICS.items():
            if key not in cur:
                v = _ncu_metric_line(line, label)
                if v is not None:
                    cur[key] = v
    _flush()
    return rows


def parse_ncu_raw(path: Path) -> list[dict]:
    """Pull the extra launch metrics ncu details omits (waves, registers).
    Keyed by row order so it can be merged onto parse_ncu_details output."""
    out: list[dict] = []
    try:
        with path.open(newline="", errors="replace") as fh:
            reader = csv.reader(fh)
            header = next(reader, [])
            units = next(reader, [])  # ncu raw csv has a units row we skip
            idx = {name: i for i, name in enumerate(header)}

            def col(row, name):
                i = idx.get(name)
                return _num(row[i]) if i is not None and i < len(row) else None

            for row in reader:
                if not row or not row[idx.get("Kernel Name", 0)].strip():
                    continue
                out.append({
                    "kernel": row[idx.get("Kernel Name", 4)][:160],
                    "waves": col(row, "launch__waves_per_multiprocessor"),
                    "regs": col(row, "launch__registers_per_thread"),
                    "grid": row[idx["Grid Size"]] if "Grid Size" in idx else "",
                    "block": row[idx["Block Size"]] if "Block Size" in idx else "",
                })
    except (OSError, StopIteration):
        return []
    return out


def merge_ncu(details: list[dict], raw: list[dict]) -> list[dict]:
    """Augment details rows with waves/registers from the matching raw row.
    Match on kernel name first; fall back to positional when names line up 1:1."""
    raw_by_name: dict[str, list[dict]] = collections.defaultdict(list)
    for r in raw:
        raw_by_name[r["kernel"]].append(r)
    for i, d in enumerate(details):
        cand = raw_by_name.get(d["kernel"])
        rr = cand.pop(0) if cand else (raw[i] if i < len(raw) and len(raw) == len(details) else None)
        if rr:
            for k in ("waves", "regs", "grid", "block"):
                if rr.get(k) is not None and rr.get(k) != "":
                    d.setdefault(k, rr[k])
    return details


def parse_ncu_log(path: Path) -> dict:
    txt = path.read_text(errors="replace")
    return {
        "file": path.name,
        "profiles": len(re.findall(r"==PROF== Profiling", txt)),
        "launchfailed": "LaunchFailed" in txt,
        "exit": (re.findall(r"EXIT=(\d+)", txt) or [""])[-1],
    }


# --------------------------------------------------------------------------- #
# 3. nsys kernel summary text exports (*_kernsum.txt) — whole-run (mostly prefill)
# --------------------------------------------------------------------------- #
def parse_nsys_kernsum(path: Path, kernel_cats=KERNEL_CATS) -> dict:
    cats: dict[str, list] = collections.defaultdict(lambda: [0.0, 0])
    top: list[dict] = []
    total_ns = 0.0
    for line in path.read_text(errors="replace").splitlines():
        parts = line.strip().split(maxsplit=8)
        if len(parts) < 9:
            continue
        try:
            share = float(parts[0])
            tot = float(parts[1].replace(",", ""))
            inst = int(parts[2].replace(",", ""))
        except ValueError:
            continue
        name = parts[8]
        c = categorize(name, kernel_cats)
        cats[c][0] += tot
        cats[c][1] += inst
        total_ns += tot
        top.append({"share": share, "count": inst, "cat": c, "name": name[:160]})
    cat_rows = {
        c: {"us": v[0] / 1e3, "pct": _pct(v[0], total_ns), "count": v[1]}
        for c, v in cats.items()
    }
    return {
        "file": path.name,
        "total_ms": total_ns / 1e6,
        "cats": cat_rows,
        "top_kernels": sorted(top, key=lambda r: -r["share"])[:40],
        "note": "whole-run (~prefill-dominated); not steady-decode attribution",
    }


# --------------------------------------------------------------------------- #
# 4. serve_load2 e2e / ceiling logs (e2e_*.txt, ceil_*.txt)
# --------------------------------------------------------------------------- #
_RESULT_RE = re.compile(
    r"\[RESULT\] conc=(?P<conc>\d+) in~(?P<inlen>\d+) out=(?P<outlen>\d+).*?"
    r"AGG output tok/s \(steady\) = (?P<agg>[0-9.]+).*?"
    r"TTFT s: p50=(?P<ttft50>[0-9.]+) p99=(?P<ttft99>[0-9.]+).*?"
    r"per-req decode tok/s: p50=(?P<dec50>[0-9.]+).*?tokens/chunk=(?P<tpc>[0-9.]+).*?"
    r"steady tokens=(?P<toks>[0-9]+) errors=(?P<errs>\d+)",
    re.S,
)

# file-stem -> (config tag, kind)
_E2E_CONFIG = {
    "e2e_mtp_off": ("mtp_off", "sweep"), "e2e_mtp_off2": ("mtp_off_r2", "sweep"),
    "e2e_mtp_on": ("mtp_on", "sweep"), "e2e_noAR": ("no_all_reduce", "sweep"),
    "e2e_nograph": ("no_cuda_graph", "sweep"),
    "ceil_base": ("ceiling_none", "ceiling"), "ceil_moe": ("ceiling_noop_moe", "ceiling"),
    "ceil_gdn": ("ceiling_noop_gdn", "ceiling"), "ceil_attn": ("ceiling_noop_attn", "ceiling"),
}


def parse_serve_log(path: Path) -> list[dict]:
    stem = path.stem
    tag, kind = _E2E_CONFIG.get(stem, (stem, "sweep"))
    rows = []
    for m in _RESULT_RE.finditer(path.read_text(errors="replace")):
        g = m.groupdict()
        rows.append({
            "file": path.name, "config": tag, "kind": kind,
            "conc": int(g["conc"]), "in_len": int(g["inlen"]), "out_len": int(g["outlen"]),
            "agg_toks": float(g["agg"]), "ttft_p50": float(g["ttft50"]), "ttft_p99": float(g["ttft99"]),
            "decode_p50": float(g["dec50"]), "tokens_per_chunk": float(g["tpc"]),
            "steady_tokens": int(g["toks"]), "errors": int(g["errs"]),
        })
    return rows


# --------------------------------------------------------------------------- #
# top-level ingest
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    meta: dict = field(default_factory=dict)
    health: dict = field(default_factory=dict)
    e2e: list = field(default_factory=list)
    decode: list = field(default_factory=list)   # torch traces
    nsys: list = field(default_factory=list)
    ncu: list = field(default_factory=list)       # one entry per ncu capture file
    findings: dict = field(default_factory=dict)  # filled by findings.py

    def to_dict(self) -> dict:
        return {
            "meta": self.meta, "health": self.health, "e2e": self.e2e,
            "decode": self.decode, "nsys": self.nsys, "ncu": self.ncu,
            "findings": self.findings, "display": DISPLAY,
        }


def ingest(datadir: Path, kernel_cats=KERNEL_CATS, use_cache: bool = True) -> Dataset:
    datadir = Path(datadir).expanduser()
    ds = Dataset()

    # torch decode traces
    for p in sorted(datadir.glob("*.trace.json.gz")):
        ds.decode.append(parse_torch_trace(p, kernel_cats, use_cache=use_cache))

    # nsys kernel summaries
    for p in sorted(datadir.glob("*_kernsum.txt")):
        ds.nsys.append(parse_nsys_kernsum(p, kernel_cats))

    # e2e + ceiling
    for p in sorted(datadir.glob("e2e_*.txt")) + sorted(datadir.glob("ceil_*.txt")):
        ds.e2e.extend(parse_serve_log(p))

    # ncu captures: group <stem>_details.txt + <stem>_raw.csv + <stem>.log
    ncu_stems = sorted({p.name[: -len("_details.txt")] for p in datadir.glob("ncu_*_details.txt")}
                       | {p.name[: -len("_raw.csv")] for p in datadir.glob("ncu_*_raw.csv")})
    for stem in ncu_stems:
        details_p = datadir / f"{stem}_details.txt"
        raw_p = datadir / f"{stem}_raw.csv"
        log_p = datadir / f"{stem}.log"
        details = parse_ncu_details(details_p, kernel_cats) if details_p.exists() else []
        raw = parse_ncu_raw(raw_p) if raw_p.exists() else []
        kernels = merge_ncu(details, raw) if details else [
            {"kernel": r["kernel"], "cat": categorize(r["kernel"], kernel_cats),
             "valid": False, **r} for r in raw
        ]
        n_valid = sum(1 for k in kernels if k.get("valid"))
        label = stem.replace("ncu_", "")
        regime = "prefill" if "prefill" in label.lower() else "decode"
        bm = re.search(r"b(\d+)", label)
        batch = int(bm.group(1)) if bm else None
        ds.ncu.append({
            "label": label,
            "regime": regime,         # decode captures characterize the decode breakdown; prefill captures don't
            "batch": batch,           # serving batch (>=32) is trustworthy; bs1 shows phantom occupancy headroom
            "details_file": details_p.name if details_p.exists() else None,
            "raw_file": raw_p.name if raw_p.exists() else None,
            "n_kernels": len(kernels),
            "n_valid": n_valid,
            "n_cluster": sum(1 for k in kernels if k.get("cluster")),
            "all_nan": len(kernels) > 0 and n_valid == 0,
            "log": parse_ncu_log(log_p) if log_p.exists() else None,
            "kernels": kernels,
        })

    # health: bogus summary.txt + nan captures + binary-only reps
    bogus_summaries = [p.name for p in datadir.glob("*_summary.txt")
                       if "==ERROR==" in p.read_text(errors="replace")[:200]]
    nan_caps = [c["label"] for c in ds.ncu if c["all_nan"]]
    ds.health = {
        "torch_traces": len(ds.decode),
        "nsys_rep_binaries": len(list(datadir.glob("*.nsys-rep"))),
        "nsys_kernsum_exports": len(ds.nsys),
        "ncu_rep_binaries": len(list(datadir.glob("*.ncu-rep"))),
        "ncu_captures_parsed": len(ds.ncu),
        "ncu_all_nan_captures": nan_caps,
        "bogus_summary_exports": bogus_summaries,
        "e2e_rows": len(ds.e2e),
        "notes": [
            "Mac-side: .nsys-rep/.ncu-rep binaries are opaque (no local NVIDIA CLI). "
            "Only their text exports are parsed.",
            f"{len(nan_caps)} ncu capture(s) came back all-NaN (cluster/CLC kernels "
            "ncu kernel-replay cannot count): " + (", ".join(nan_caps) or "none"),
            f"{len(bogus_summaries)} _summary.txt export(s) are ==ERROR== (wrong --page) "
            "and were ignored.",
        ],
    }
    ds.meta = {
        "datadir": str(datadir),
        "n_files": len(list(datadir.iterdir())),
        "schema_version": 1,
    }
    return ds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("datadir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="write dataset JSON here (default: stdout summary only)")
    args = ap.parse_args()
    ds = ingest(args.datadir)
    d = ds.to_dict()
    if args.out:
        args.out.write_text(json.dumps(d, indent=2))
        print(f"wrote {args.out}")
    print(json.dumps(d["health"], indent=2))


if __name__ == "__main__":
    main()
