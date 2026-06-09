#!/usr/bin/env python3
"""Tests for the profiler pipeline.

These use tiny synthetic fixtures (no GPU, no real artifacts) and deliberately
encode the anti-phantom-win guarantees so they can't silently regress:
  * a prefill capture must NOT characterize a decode category;
  * a cluster / CLC kernel must NOT be reported as a fusion win;
  * a bs=1 capture must NOT be preferred over the serving-batch capture;
  * the Amdahl ceiling math is exact.

Run:  python3 -m pytest tools/profiler/tests/ -q
   or: python3 tools/profiler/tests/test_profiler.py   (no pytest needed)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import ingest          # noqa: E402
import findings as fnd  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _write_details(p: Path, blocks: list[tuple[str, dict, bool]]):
    """blocks: (kernel_name, {metric_label: value}, clc_flag)."""
    lines = ["[123] python@host"]
    for name, metrics, clc in blocks:
        lines.append(f"  {name} (32, 1, 1)x(256, 1, 1), Context 1, Stream 7, Device 0, CC 10.3")
        if clc:
            lines.append("    Warning: The result was collected with the Work ID/Cluster Launch Control (CLC) feature enabled.")
        lines.append("    Section: GPU Speed Of Light Throughput")
        for label, val in metrics.items():
            lines.append(f"    {label}    %    {val}")
        lines.append("    Section: Occupancy")
    p.write_text("\n".join(lines))


def _make_dataset(tmp: Path):
    # decode @ bs32: FP4 GEMM is memory-bound (the truth). NOTE clc=True — the FP4
    # cutlass GEMM uses 2cta thread-block clusters, so ncu emits the CLC warning,
    # but its 80% DRAM reading is valid. It must NOT be excluded as "cluster".
    _write_details(tmp / "ncu_fp4gemm_b32_details.txt", [
        ("bmm_E2m1_E2m1E2m1_t128x32x512", {"Compute (SM) Throughput": 49, "Memory Throughput": 80, "DRAM Throughput": 80, "Achieved Occupancy": 25}, True),
    ])
    # decode @ bs1: same kernel reads LOW dram (the phantom) — must be ignored in favour of bs32
    _write_details(tmp / "ncu_fp4gemm_b1_details.txt", [
        ("bmm_E2m1_E2m1E2m1_t128x8x512", {"Compute (SM) Throughput": 14, "Memory Throughput": 22, "DRAM Throughput": 22, "Achieved Occupancy": 20}, True),
    ])
    # PREFILL: same FP4 family but big-M => compute-bound. Must NOT speak for decode.
    _write_details(tmp / "ncu_prefill_details.txt", [
        ("bmm_E2m1_E2m1E2m1_t128x256x512", {"Compute (SM) Throughput": 82, "Memory Throughput": 30, "DRAM Throughput": 30, "Achieved Occupancy": 60}, False),
    ])
    # decode glue: a CLUSTER routing kernel (CLC) reads fake-low util -> must not be a "fuse" win
    _write_details(tmp / "ncu_glue_gdn_details.txt", [
        ("void routingIndicesClusterKernel<KernelParams<512,16>>", {"Compute (SM) Throughput": 4, "Memory Throughput": 6, "DRAM Throughput": 6, "Achieved Occupancy": 12}, True),
        ("void moe::finalizeKernelVecLoad<T>", {"Compute (SM) Throughput": 6, "Memory Throughput": 7, "DRAM Throughput": 7, "Achieved Occupancy": 12}, False),
    ])

    # torch decode trace (mtp_off, TP0)
    events = []

    def kern(name, dur, n):
        for _ in range(n):
            events.append({"ph": "X", "cat": "kernel", "name": name, "dur": dur, "ts": 0})

    kern("bmm_E2m1_E2m1E2m1_t128x32x512_decode", 100, 39)   # 39% FP4 MoE GEMM (memory floor)
    kern("nvjet_sm103_gemm", 100, 25)                       # 25% dense GEMM
    kern("void routingIndicesClusterKernel<...>", 100, 5)   # 5% routing (cluster floor)
    kern("void moe::finalizeKernelVecLoad<T>", 100, 2)      # 2% finalize (fuse win)
    kern("act_and_mul_kernel", 100, 1)                      # 1% act (no ncu -> unknown)
    with gzip.open(tmp / "run.1234-TP-0.trace.json.gz", "wt") as fh:
        json.dump({"traceEvents": events}, fh)

    # e2e sweeps
    (tmp / "e2e_mtp_off.txt").write_text(_serve(64, 2800) + _serve(32, 1500))
    (tmp / "e2e_mtp_on.txt").write_text(_serve(64, 2400) + _serve(32, 1900))
    (tmp / "e2e_nograph.txt").write_text(_serve(64, 620) + _serve(32, 640))
    (tmp / "e2e_noAR.txt").write_text(_serve(64, 2950))   # ablation: must NOT be the reported peak
    # a bogus ncu summary export (wrong --page)
    (tmp / "ncu_fp4gemm_b32_summary.txt").write_text("==ERROR== the argument for option '--page' is invalid.")


def _serve(conc, agg):
    return (f"----- conc={conc} -----\n[RESULT] conc={conc} in~16384 out=1024 dur=45s steady=23s\n"
            f"  AGG output tok/s (steady) = {agg}   per-stream = {agg/conc:.1f}\n"
            f"  TTFT s: p50=1.00 p99=2.00 (n=8)\n"
            f"  per-req decode tok/s: p50=40.0   tokens/chunk=1.00\n"
            f"  steady tokens=1000 errors=0\nSERVE_LOAD2_DONE\n")


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def _run(tmp: Path):
    ds = ingest.ingest(tmp).to_dict()
    return ds, fnd.derive(ds)


def test_clc_cluster_detection(tmp):
    ds, _ = _run(tmp)
    caps = {c["label"]: c for c in ds["ncu"]}
    routing = [k for k in caps["glue_gdn"]["kernels"] if "routing" in k["kernel"].lower()][0]
    assert routing["clc"] is True and routing["cluster"] is True, "CLC/cluster routing kernel not flagged"
    finalize = [k for k in caps["glue_gdn"]["kernels"] if "finalize" in k["kernel"].lower()][0]
    assert finalize["cluster"] is False, "finalize wrongly flagged as cluster"
    # the separation that fixed the FP4 mislabel: CLC warning != structural cluster
    fp4 = caps["fp4gemm_b32"]["kernels"][0]
    assert fp4["clc"] is True and fp4["cluster"] is False, "FP4 GEMM (clc, not name-cluster) misflagged"


def test_regime_and_batch_tags(tmp):
    ds, _ = _run(tmp)
    caps = {c["label"]: c for c in ds["ncu"]}
    assert caps["prefill"]["regime"] == "prefill"
    assert caps["fp4gemm_b32"]["regime"] == "decode" and caps["fp4gemm_b32"]["batch"] == 32
    assert caps["fp4gemm_b1"]["batch"] == 1


def test_fp4_is_memory_bound_from_decode_not_prefill(tmp):
    """The headline correctness check: decode FP4 GEMM must read MEMORY-bound
    (bs32, 80% dram), never compute-bound (which is the prefill capture)."""
    _, f = _run(tmp)
    fp4 = [c for c in f["decode_canonical"]["categories"] if c["cat"] == "fp4_moe_gemm"][0]
    assert fp4["bound_type"] == "memory", f"expected memory-bound, got {fp4['bound_type']}"
    assert fp4["winnable"] is False
    assert fp4["ncu"]["capture"] == "fp4gemm_b32", "did not prefer serving-batch capture"


def test_routing_is_not_a_fusion_win(tmp):
    """The phantom-win guard: a cluster/CLC kernel must never be 'winnable'."""
    _, f = _run(tmp)
    routing = [c for c in f["decode_canonical"]["categories"] if c["cat"] == "moe_routing"][0]
    assert routing["winnable"] is False, "routing cluster kernel reported as winnable!"
    assert routing["bound_type"] == "cluster"
    assert not any("routing" in o["category"] for o in f["opportunities"] if o["est_decode_gain_pct"]), \
        "routing appears as a fusion opportunity"


def test_finalize_is_a_fusion_win(tmp):
    _, f = _run(tmp)
    fin = [c for c in f["decode_canonical"]["categories"] if c["cat"] == "moe_finalize"][0]
    assert fin["winnable"] is True and fin["bound_type"] == "latency"


def test_peak_is_primary_not_ablation(tmp):
    _, f = _run(tmp)
    assert f["peak"]["config"] in ("mtp_off", "mtp_on"), f"peak picked an ablation: {f['peak']}"
    assert f["peak"]["tok_s"] == 2800


def test_amdahl_ceiling_math(tmp):
    _, f = _run(tmp)
    a = f["amdahl"]
    # winnable categories here: finalize(2%) only (act is unknown w/o ncu). floor: fp4 39 + dense? dense has no ncu -> unknown.
    # exact ceiling = 1/(1 - winnable/100)
    expected = round(1.0 / (1.0 - a["winnable_pct"] / 100.0), 3)
    assert a["max_decode_speedup_if_winnable_eliminated"] == expected
    assert a["winnable_pct"] + a["floor_pct"] + a["unknown_pct"] <= 100.01


def test_bogus_summary_flagged(tmp):
    ds, _ = _run(tmp)
    assert any("summary" in s for s in ds["health"]["bogus_summary_exports"])


def test_trace_cache_roundtrips(tmp):
    """Second parse must hit the cache and return an identical summary."""
    trace = next(tmp.glob("*.trace.json.gz"))
    a = ingest.parse_torch_trace(trace, use_cache=True)
    assert ingest._trace_cache_path(trace).exists(), "cache file not written"
    b = ingest.parse_torch_trace(trace, use_cache=True)
    assert a == b, "cached parse differs from fresh parse"
    # a no-cache parse must still match
    assert ingest.parse_torch_trace(trace, use_cache=False) == a


def main():
    import tempfile
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            _make_dataset(tmp)
            try:
                t(tmp)
                print(f"  PASS  {t.__name__}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL  {t.__name__}: {e}")
                raise
    print(f"\n{passed}/{len(tests)} passed")


# pytest fixture
try:
    import pytest

    @pytest.fixture
    def tmp(tmp_path):
        _make_dataset(tmp_path)
        return tmp_path
except ImportError:
    pass


if __name__ == "__main__":
    main()
