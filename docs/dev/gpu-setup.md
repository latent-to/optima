# GPU setup

This page prepares a contributor or maintainer GPU host for component verification and
performance development. It is not a production arena deployment recipe. Production
candidate execution additionally requires the isolated worker and injected
arena boundaries in the [validator guide](../validator-guide/first-hour.md).

## Prerequisites

- Linux with a supported NVIDIA GPU and driver;
- a CUDA toolkit compatible with the installed PyTorch/SGLang build, including
  `nvcc` for native contribution builds;
- enough GPU memory for the chosen model and tensor-parallel topology;
- Python 3.10 or newer;
- container runtime and NVIDIA device integration for production-shaped tests;
- `ninja` and a compiler toolchain where reviewed native builds require them.

Confirm the host before installing Python packages:

```bash
nvidia-smi
nvcc --version
python3 --version
```

## Install the pinned runtime

Torch is intentionally not pinned by Optima's base package because its wheel
must match the host CUDA/runtime environment. Resolve SGLang and its Torch
family for the host first, then install Optima without replacing that stack:

```bash
git clone https://github.com/latent-to/cacheon.git
cd cacheon

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip uv

# General resolver path; verify that the selected Torch wheel matches this host.
uv pip install "sglang==0.5.13.post1" ninja datasets
uv pip install -e ".[dev,release]"
```

CUDA 13 hosts may require the CUDA-specific Torch index and a prerelease
dependency allowed by SGLang. On such a host, replace the first `uv pip
install` above with the deployment-reviewed equivalent of:

```bash
uv pip install --prerelease=allow --torch-backend=cu130 \
  "sglang==0.5.13.post1" ninja datasets
```

Do not use `--torch-backend=cu130` on a non-CUDA-13 host. In either path,
record `python -c 'import torch; print(torch.__version__, torch.version.cuda)'`
and reject the environment if the resolved wheel does not match the driver,
toolkit, and deployment lock.

The repository's current SGLang contract is `0.5.13.post1`, but installing the
pin is not evidence that its GPU gates passed. Check the dated validation
boundary in [State of record](../reference/state-of-record.md) and the proof
procedure in [SGLang compatibility](sglang-tracking.md). A deployment lockfile
or image is stronger authority than this illustrative installation sequence;
do not let a package resolver silently replace its Torch/CUDA stack.

Set toolchain variables only when the host needs them:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Architecture variables belong to the exact GPU and build policy. Do not copy a
Hopper, B200, or B300 architecture value from another machine without checking
the device and compiler support.

## Freeze the environment you are testing

Before recording a result, capture enough identity to reproduce the execution:

- host GPU model/count and logical visibility;
- driver, CUDA runtime/toolkit, and `nvcc` versions;
- Torch, SGLang, Triton, FlashInfer/CUTLASS, and Optima revisions;
- container/base-image digest where applicable;
- model revision, manifest, and content digests;
- tensor-parallel world size and topology class;
- CUDA-graph state, dtype, workload/seed identity, and power/clock policy.

Package names alone are not sufficient. Two hosts can report the same SGLang
version while loading different Torch/CUDA or native products. For formal
evaluation and release work, use the typed arena, native-build, model, and
release identities rather than a pasted `pip freeze` as authority.

## Preflight

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

python -m optima.cli compat
```

`compat` reports the installed SGLang version and checks its imports and
signatures. A pin mismatch marks the version row `DIFFERS from pin`, fails that
row, and makes the command exit nonzero. The canary is necessary after
installation or an upgrade, but it cannot prove graph capture, model load,
distributed topology, numerical fidelity, or performance.

If this host also has the deployment-approved Bittensor SDK installed, run its
independent import/signature canary with `python -m optima.cli chain-compat`.

Treat the preflight as three independent questions:

1. **Device readiness:** does Torch see the intended devices and capability?
2. **Static compatibility:** are the registered SGLang symbols/signatures
   present at the exact pin?
3. **Behavioral readiness:** do faithful and broken controls, graph replay,
   model load, and the real topology behave as expected?

Only the first two are covered by the commands above.

## Verify a contribution

Start with a known example, then the contribution:

```bash
python -m optima.cli scan examples/miner_silu_triton
python -m optima.cli verify examples/miner_silu_triton \
  --device cuda --dtype bfloat16

python -m optima.cli scan path/to/bundle
python -m optima.cli verify path/to/bundle \
  --device cuda --dtype bfloat16 --model <registered-model-key>
```

Collective targets need the arena's real world size and homogeneous visible
devices:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m optima.cli verify path/to/collective-bundle \
  --device cuda --world-size 4
```

CUDA verification captures applicable entries, refreshes registered dynamic
inputs, poisons outputs, and checks multiple replays. It remains a component
gate, not a serving crown.

For collectives, “the command returned” is not enough. Check that every rank
selected the same candidate, received the validator-owned group, activated the
registered seam, completed it, and produced the expected output. A rank-local
fallback would diverge the collective, so the candidate engine must abort
rather than continue with mixed implementations.

## Complete-engine performance development

Optima deliberately exposes no local qualification command. Contributors may profile
and A/B the complete serving engine on a disposable host appropriate for candidate code,
using the published arena contract as the environment specification. Keep the model,
runtime, topology, graph mode, workload, and charged-work basis fixed; measure the
candidate between two incumbent runs and distrust a delta smaller than baseline drift.
The canonical [performance-development procedure](../validator-guide/running-evals.md#performance-development)
defines the required inputs and local result record; no repository command materializes
this complete-engine bracket.

Production v3 qualification materializes the exact incumbent and candidate engines
through an injected arena service, loads each once onto a disjoint resident TP lane,
and serializes B/C/B′ plus policy-authorized C′/B″ reads. It then runs registered
eager audit A, tears down candidate lifetimes, and obtains candidate-free pristine T
quality evidence. A contributor-controlled model run cannot substitute for that
authority.

## From component proof to arena proof

Move upward only after the lower layer is green:

| Layer | Required observation | Still does not prove |
|---|---|---|
| Component `verify` | Registered reference and graph replay for exercised cases | Model integration or speedup |
| Local complete-engine A/B | Model can load and the selected delta can improve the matched workload | Validator isolation, hidden quality, crown authority, independent reproduction |
| Arena screen | Static/build/ABI/graph/abbreviated-serving gates all promote | B/C/B′ drift, T quality, settlement |
| Qualification PASS | Exact marginal complete-engine delta clears all registered gates | Crown until independent reproduction |
| Two matching PASSes | Candidate is eligible for cohort settlement; the current registered cohort winner may be crowned while another valid pair is held | Integration safety or release readiness |

Keep local A/B results as engineering evidence, labeled with their exact environment and
denominator. Do not treat them as qualification evidence.

## Common failures

| Symptom | Check |
|---|---|
| Torch cannot see CUDA | Driver, wheel CUDA version, container device wiring |
| `nvcc` is missing | Install/mount the matching toolkit; set `CUDA_HOME` |
| `compat` reports a moved seam | Confirm exact SGLang pin; follow the bump process rather than patching around it |
| Scheduler cannot import Optima | Install editable package in the same environment and use module invocation |
| Collective hangs | Rank/world-size agreement, visible devices, topology, and clean prior processes |
| Graph path falls back | Capability metadata, `graph_safe`, static allocations, host syncs, dynamic-input contract |
| Baselines drift | Stop scoring; inspect thermals, clocks, competing processes, device cleanup, and arena conditioning |

Source: [`optima/compat.py`](https://github.com/latent-to/cacheon/blob/main/optima/compat.py),
[`optima/verify.py`](https://github.com/latent-to/cacheon/blob/main/optima/verify.py), and
[`optima/verify_collective.py`](https://github.com/latent-to/cacheon/blob/main/optima/verify_collective.py).
