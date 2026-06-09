# Optima profiler — dataset → findings → dashboard

Feed it a directory of GPU-profiling artifacts; get back **ranked, defensible
findings** and a **self-contained HTML dashboard**. Built for the Optima loop:
profile a model (or *our* patches), and answer *where is the win, how big can it
possibly be, and where is there no win* — without manufacturing phantom wins.

```bash
python3 tools/profiler/build.py ~/Downloads/github/temp/profiles_b300 -o profiler_out
open profiler_out/report.html      # offline, zero-dependency, double-click to view
```

That writes three files into `-o` (default `profiler_out/`):
- `report.html` — the dashboard (all data inlined; opens with no server, no deps).
- `dataset.json` — the normalized dataset (programmatic use / diffing runs).
- `findings.json` — the derived verdicts.

No third-party Python, no NVIDIA CLIs, no network. Pure stdlib — because the Mac
checkout has none of those.

## What it ingests

Drop any mix of these into the dataset dir (it reads what's there, flags what's
missing or junk):

| artifact | what it gives |
|---|---|
| `*.trace.json.gz` (torch profiler) | the **clean decode** kernel-time breakdown by category |
| `ncu_*_details.txt` + `ncu_*_raw.csv` | per-kernel **bound-type** (compute/memory/latency) + waves/registers |
| `ncu_*.log` | capture health (profiles, LaunchFailed, exit) |
| `*_kernsum.txt` (nsys `cuda_gpu_kern_sum`) | whole-run (≈prefill) kernel shares |
| `e2e_*.txt`, `ceil_*.txt` (serve_load2) | throughput vs concurrency + A/B levers |

**Binaries (`.nsys-rep` / `.ncu-rep`) are opaque on a Mac** — export their text
on the GPU box first (`ncu --import x.ncu-rep --page details > x_details.txt` and
`--page raw --csv > x_raw.csv`; `nsys stats --report cuda_gpu_kern_sum`). The
report's *Artifact health* panel lists what was binary-only so you know to export it.

## The insight engine (`findings.py`)

The core join: decode kernel-time **share** (torch) × bound-type **verdict**
(ncu) → per-category **winnability**, then an **Amdahl ceiling** on realistic
end-to-end gain. Everything is derived from the raw numbers; the thresholds are
explicit constants at the top of `findings.py`, not magic — argue with them.

### Anti-phantom-win guards (learned the hard way; encoded as tests)

These are the traps that produced false "wins" before. The engine refuses them:

1. **Regime separation.** A *prefill* big-M GEMM is compute-bound; the *decode*
   skinny-M GEMM of the same name is memory-bound. A decode category is only ever
   characterized by a **decode-regime** ncu capture — never a prefill one.
2. **Serving batch, not bs=1.** A bs=1 capture shows phantom occupancy "headroom"
   that vanishes at the serving batch. The engine de-prioritizes bs=1 captures.
3. **Cluster ≠ CLC.** A *structural* cluster kernel (e.g.
   `routingIndicesClusterKernel`) is vendor MoE-dispatch you can't fuse → never a
   win. The CLC *warning* (on a `2cta` FP4 cutlass GEMM) only corrupts
   occupancy/launch-counts — **throughput stays valid**, so an 80%-DRAM kernel is
   still correctly a memory-bound floor, not excluded.
4. **Only ncu-proven wins count.** A category is "winnable" only with a clean
   decode ncu capture showing it latency-bound. Everything else is flagged
   **"PROFILE IT"**, not optimistically assumed fusable.
5. **Honest peak.** Headline throughput is the best **primary** config
   (mtp on/off), never an ablation (`--disable-*`).
6. **Junk is flagged, not averaged.** All-NaN captures (CLC kernels ncu can't
   count) and `==ERROR==` `_summary.txt` exports are surfaced in *Artifact health*.

## The dashboard

`report.html` sections: **Findings** (Amdahl winnable/floor/unknown split,
ranked opportunities, hard constraints) · **E2E throughput** (tok/s vs
concurrency) · **Decode breakdown** (categories colored by verdict) · **NCU
bound-types** (per-kernel table) · **Kernel explorer** (search/sort every top
kernel) · **Config levers** (CUDA-graph / MTP / all-reduce / ceiling) ·
**Artifact health**.

## Extending to a new model

The kernel taxonomy is a config: `KERNEL_CATS` (regex → category) and `DISPLAY`
in `ingest.py`. Add patterns for the new model's kernel names; everything
downstream (findings, dashboard) adapts. ncu capture labels drive regime/batch
detection: a label containing `prefill` → prefill regime; `b32`/`b1` → batch.

## Modules
- `ingest.py` — artifact parsers → normalized dataset (CLI: `python3 ingest.py <dir> -o dataset.json`).
- `findings.py` — dataset → verdicts (CLI: `python3 findings.py dataset.json`).
- `report.py` — dataset + findings → `report.html` (template in `templates/`).
- `build.py` — the one-command pipeline.
- `tests/test_profiler.py` — synthetic fixtures encoding the guards above
  (`python3 tools/profiler/tests/test_profiler.py`, no pytest needed).

> Note: torch traces are large (tens of MB gzipped → ~1 GB JSON each); parsing
> all of them is the slow step (~40 s, single-threaded stdlib `json`). The ncu /
> nsys / e2e parsing is instant.
