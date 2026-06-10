# Optima profiler — dataset → findings → dashboard

Feed it a directory of GPU-profiling artifacts; get back **ranked, defensible
findings** and a **self-contained HTML dashboard**. Built for the Optima loop:
profile a model (or *our* patches), and answer *where is the win, how big can it
possibly be, and where is there no win* — without manufacturing phantom wins.

```bash
python3 tools/profiler/build.py ~/Downloads/github/temp/profiles_b300 -o profiler_out
open profiler_out/report.html      # offline, zero-dependency, double-click to view
python3 tools/profiler/plan.py profiler_out/dataset.json -o profiler_out/capture_plan.md
```

That writes three files into `-o` (default `profiler_out/`):
- `report.html` — the dashboard (all data inlined; opens with no server, no deps).
- `dataset.json` — the normalized dataset (programmatic use / diffing runs).
- `findings.json` — the derived verdicts.

`plan.py` adds the next operational layer: it turns a report into a bounded
capture plan (`capture_plan.md`) with concrete backend A/Bs and `ncu` target
rows. Use it to avoid hand-editing a fresh profiler battery for every new model.

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

## Workflow: how this fits the loop

This tool is the **analysis** half. The **capture** half is
`experiments/qwen35/pod_scripts/` (`profile_run.sh` on the GPU box →
`mac_pull.sh` down). They're designed to fit: `profile_run.sh` produces exactly
the artifacts this tool ingests. The loop is always the same:

```
capture on pod  →  pull to Mac  →  build.py <dir>  →  open report.html
```

### The profile pack, not a profiling loop

There is no single profiler invocation that gives every answer. `nsys` gives the
timeline and attribution, `ncu` gives per-kernel counters and supports
multi-process/multi-GPU profiling, but replay mode must match the workload
(kernel/range/application replay, communicator/lockstep for mandatory-concurrent
communication kernels). e2e serving is the only truth for throughput. Treat one
**profile pack** as the unit:

1. **Serving truth**: e2e sweep at the target concurrencies, plus a short torch
   steady-decode trace from the same server config.
2. **Timeline**: `nsys` decode at serving batch, bs1 control, max batch, prefill,
   and the long-context point you care about; always export `cuda_gpu_kern_sum`
   and `cuda_gpu_trace` on the GPU box.
3. **Counters**: `ncu` only the categories from the report/plan at serving batch
   plus bs1 where useful. TP=1 is a convenience for isolated non-collective
   kernels when the model fits; TP>1 is valid and required for communication or
   sharded-kernel truth.
4. **Communication counters**: for NCCL/NVSHMEM/all-reduce-style work, use
   `--target-processes all`, the appropriate `--communicator` mode, lockstep
   launch, and NVTX/range/application replay rather than pretending a TP=1
   capture answers the question.
5. **Small backend matrix**: run only pre-declared flag A/Bs that the logs or
   model code justify, e.g. `--linear-attn-decode-backend flashinfer` vs
   `--linear-attn-decode-backend triton` for Qwen GDN.

After `build.py`, run:

```bash
python3 tools/profiler/plan.py profiler_out/dataset.json -o profiler_out/capture_plan.md
```

If `capture_plan.md` is empty except stop conditions, stop profiling and
optimize. If it lists rows, run only those rows. This keeps Qwen/Nemotron/Minimax
profiling bounded: first pack finds the map, the generated plan closes the grey
surface, then code work starts.

### A) New model on a new pod (e.g. Nemotron)

1. **Capture** with `pod_scripts/profile_run.sh` (the V2 driver): point it at the
   new model + serve args. Keep capture **labels** consistent — the analyzer reads
   regime/batch from them: a label containing `prefill` → prefill regime;
   `b32`/`b1` → that batch. (So `ncu_fp4gemm_b32`, `ncu_prefill`, etc.)
2. **Pull** with `mac_pull.sh` → `~/Downloads/.../profiles_nemotron`.
3. **Teach it the new kernels** — the only model-specific step: add regexes to
   `KERNEL_CATS` in `ingest.py` for the new model's kernels (Mamba-2 SSD, NVFP4
   LatentMoE, etc.). Everything downstream (findings, dashboard) adapts.
4. **Analyze**: `python3 tools/profiler/build.py ~/Downloads/.../profiles_nemotron
   -o nemotron_out` → open `nemotron_out/report.html`. You immediately get the
   winnable surface, the vendor floors, the Amdahl ceiling, and the config levers.
5. **Plan closure**: `python3 tools/profiler/plan.py nemotron_out/dataset.json -o
   nemotron_out/capture_plan.md`. Copy only the generated NCU rows/A-Bs into the
   next pod pass. Do not invent extra rows unless the report names a category the
   taxonomy cannot yet classify.

### B) Measuring your OWN fused kernels (the win/no-win question)

The honest question for a patch is **baseline vs patched**, so capture two
datasets and compare:

```bash
# baseline
profile_run.sh all                         # -> profiles_base/
# with your seam/kernel active (see AGENTS.md OPTIMA_*_SEAM env vars)
OPTIMA_MOE_SEAM=1 profile_run.sh all        # -> profiles_patched/
build.py profiles_base    -o out_base
build.py profiles_patched -o out_patched
```

Then the **discipline loop** (ceiling-gate → ncu-confirm → rewrite → e2e-prove)
maps onto the two reports:

- **Before you write the kernel** — `out_base/report.html` tells you whether the
  target is worth it: its **decode share** (Amdahl — is the ceiling more than
  noise?) and its **ncu bound-type** (latency-bound = fusable; memory/compute
  floor = don't bother).
- **After you write it** — `out_patched` should show three things move together:
  (1) the glue category's **launch count drops** (you eliminated a kernel
  boundary), (2) its **decode share shrinks**, (3) **e2e tok/s rises**. If decode
  share didn't move or e2e didn't rise → it's a phantom win. Discard it.

Don't eyeball two reports — run **`compare.py`**, which arbitrates in one line:

```bash
python3 tools/profiler/compare.py out_base/dataset.json out_patched/dataset.json \
    --label-a base --label-b moe_fused -o compare.html
```

It prints the verdict + the e2e and decode-structure deltas. **E2E is the
arbiter** (steady-state, directly comparable; kernel-time *share* is not — the
denominator shifts when you fuse). **A launch-count drop is the fusion proof.**
And a delta **within ±noise is NOT a win** (default ±2%, `--noise-pct` to change)
— it says INCONCLUSIVE and tells you to re-run interleaved + clock-locked, so a
clock/thermal wobble never reads as a win. The verdicts:
- `WIN` — e2e up past noise **and** a glue category's launches collapsed.
- `APPARENT WIN` — e2e up but no structural change → suspect clock noise, confirm.
- `INCONCLUSIVE` — e2e within ±noise. Not a win.
- `REGRESSION` — e2e down past noise.

**On the nsys-only box (no ncu):** the tool still works — categories just read
"unknown / PROFILE IT" instead of a bound-type. That's fine for *proving* a
fusion: ncu is for *deciding where to fuse* (diagnose the bound-type once, on an
ncu box); nsys + e2e + the kernel-launch-count drop is enough to *prove the
fusion worked*. So: diagnose bound-types on an ncu box once per model, then
iterate on patches against the nsys-only box using e2e + launch counts.

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
- `plan.py` — `dataset.json` → bounded next-capture plan (`capture_plan.md`).
- `compare.py` — A/B two `dataset.json` → one win/no-win verdict + `compare.html`
  (CLI: `python3 compare.py base/dataset.json patched/dataset.json -o compare.html`).
- `tests/test_profiler.py` — synthetic fixtures encoding the guards above
  (`python3 tools/profiler/tests/test_profiler.py`, no pytest needed).

> Note: torch traces are large (tens of MB gzipped → ~1 GB JSON each); parsing
> all of them is the slow step (~40 s, single-threaded stdlib `json`). The ncu /
> nsys / e2e parsing is instant.
