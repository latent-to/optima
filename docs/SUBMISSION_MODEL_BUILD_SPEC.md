# Submission-model build spec (the kickstart)

> The file-by-file plan to take Optima from the n=1 swigluoai win to a subnet that ingests
> portable kernel wins — grounded in the current code (`optima/` on `docs/submission-model`,
> verified at the cited line numbers). Companion to [`SUBMISSION_MODEL.md`](SUBMISSION_MODEL.md)
> (the *why*) and [`SLOT_CONTRACT.md`](SLOT_CONTRACT.md) (the four invariants).
>
> **Still design, not code.** Snippets below are interface contracts (signatures / dataclass
> fields / manifest shapes), not finished diffs.

## Four milestones, in dependency order

| # | What | Unblocks | Size |
|---|---|---|---|
| **M0** | Re-point the MoE seam `FusedMoE.forward` → `forward_impl` | *anything* scoring under piecewise graphs | ~30 LoC + canary |
| **M1** | The epilogue override-point (+ `optima_kernels` base + per-model reference) | the first clean portable win (swigluoai) | the keystone |
| **M2** | `optima_kernels` package boundary + blessed-base lockfile + scanner allowlist | portability + cross-validator consensus | medium |
| **M3** | Second win: `attention.msa_block_score` slot + `topk_overlap` mode | proves generalization to a *different* win shape | medium |

## Verify-on-`PINNED_SGLANG` checklist (known-unknowns)

The reconstruction read sglang `pr27944-current` + flashinfer 0.6.13 clones, **not** the scored
`PINNED_SGLANG` (0.5.13.post1, `compat.py:23`). Before building, confirm against the pin:

1. **`FusedMoE.forward_impl` exists and its signature.** The dispatcher assumes `(self,
   hidden_states, topk_output)` (matches `forward`). If `forward_impl` takes extra args, M0's
   dispatcher signature must match.
2. **The piecewise-capture detector.** `dispatch.py:422` imports `is_in_piecewise_cuda_graph`
   from `sglang.srt.compilation.piecewise_context_manager`; the reconstruction cited
   `is_in_tc_piecewise_cuda_graph()` from `context_manager.py`. Confirm the exact symbol on the
   pin **and** that it reads True during graph *replay*, not only capture (the gate depends on it).
3. **The reduce site.** Confirm the TP all-reduce lives inside `forward_impl` (so patching it,
   `_run_moe_kernel` owning the reduce decision, stays faithful — `dispatch.py:508-513`).
4. **The flashinfer base-kernel version to vendor** (M1) — pin one (the win used 0.6.12; clone is
   0.6.13). Anchor by the `acc_vec_up_alpha * silu_f32(` marker, not a line number.

Each is a one-line `grep`/signature check; none is expected to block, but a wrong assumption here
silently scores the stock path (the exact bug M0 fixes).

---

## M0 — Re-point the MoE seam to `forward_impl`

**Why.** The seam patches `FusedMoE.forward` (`seams.py:46`, `integrations/sglang_moe.py:46-48`).
Under piecewise CUDA-graph capture, sglang's `forward` routes to a registered custom op that calls
`self.forward_impl(...)` **directly**, bypassing the patched `forward`. So an Optima-hosted MoE
kernel silently does not run in the production piecewise regime. `forward_impl` is the true waist —
eager, full-graph decode, and both piecewise custom-op paths converge there. This is the concrete
form of the slot-waist-ceiling "no kernel beats sglang is structural" bug.

**Changes:**

1. `optima/seams.py:46` — chokepoint string `"FusedMoE.forward"` → `"FusedMoE.forward_impl"`
   (drives the canary + docs; the table is the single source of truth).
2. `optima/integrations/sglang_moe.py` — patch `forward_impl` instead of `forward`:
   ```python
   orig_impl = FusedMoE.forward_impl
   FusedMoE.forward_impl = make_moe_dispatcher(orig_impl, registry=registry)
   FusedMoE._optima_orig_forward_impl = orig_impl
   ```
   Guard with `if not hasattr(FusedMoE, "forward_impl"): return` (fail-safe on an older pin).
   Update `uninstall()` / `is_installed()` symmetrically.
3. `optima/dispatch.py:226-299` — `make_moe_dispatcher`'s `dispatched(self, hidden_states,
   topk_output)` body is **unchanged** (it already calls `baseline_forward(self, hidden_states,
   topk_output)` and replays the reduce in `_run_moe_kernel`). Only confirm the signature matches
   `forward_impl` (checklist #1). The `_in_cuda_graph()` gate (`dispatch.py:282`) now does real
   work: a `graph_safe` kernel runs in-graph; a non-graph-safe one falls back to the original
   `forward_impl` in-graph.
4. `optima/compat.py` — the bespoke `FusedMoE.forward` signature assert (`~:114-118`) → check
   `forward_impl` has `{hidden_states, topk_output}`. (The table-driven existence check at
   `:58-69` updates automatically from `seams.py`.)

**Acceptance:** a registered `graph_safe` MoE kernel fires under piecewise capture (not only
full-capture decode). Test: mock `is_in_piecewise_cuda_graph()=True`, drive the custom-op path,
assert the dispatcher ran. GPU smoke: the broken-bundle gate still FAILs, a faithful kernel PASSes,
graphs-ON. (Same as the attention re-point to the `unified_attention_with_output` custom-op — defer
to M3, since the `attention.decode` seam is eager-only/opt-in today and isn't scored under graphs.)

---

## M1 — The epilogue override-point (the keystone)

The swigluoai win is a ~14-line GEMM1 epilogue against a kernel the miner didn't write. M1 makes
that the *unit of submission*: the validator owns a hooked base kernel; the miner ships only the
epilogue device fn + a torch reference. The mechanism is **CUTLASS EFC** (Epilogue Fusion
Customization) — already shipped in flashinfer (`epilogue_op: cutlass.Constexpr`) and NVIDIA's
CUTLASS examples. Composition happens at **load time**, so the runtime path is the *existing* MoE
dispatcher (with M0's fix) — the override submission inherits output-ownership, eligibility, quant
pairing, graph-safety, and fallback for free. (We do **not** add a `FusedOpPool` registry seam —
that would bypass the dispatcher's invariant machinery.)

### M1.1 — Manifest fields (`optima/manifest.py`)

Add two optional fields to `OpEntry` (`:62-72`) and parse them (`:178-193`):
```python
base_kernel: str | None = None     # names a validator-owned base in optima_kernels
override_point: str | None = None  # e.g. "gemm1_epilogue"
```
Add both to the `known` set (`:178`). Validation: if `override_point` is set, `base_kernel` must be
too, and `base_kernel` must resolve in `optima_kernels` (checked at load, not parse). When set,
`entry` names the override device fn (still an identifier); `prepare` is **omitted** (the validator
owns the NVFP4 weight-prep — see M1.4).

### M1.2 — `optima_kernels` package (new, top-level, ZERO sglang imports)

```
optima_kernels/
  __init__.py
  override.py          # OVERRIDE_POINTS registry + compose()
  moe/nvfp4_megakernel.py   # vendored-once flashinfer base, epilogue hooked
  codec/nvfp4.py       # re-homed-clean quantize / swizzle / mma-convert / alpha
```

- **`moe/nvfp4_megakernel.py`** — vendor flashinfer's
  `blockscaled_contiguous_gather_grouped_gemm_swiglu_fusion` **once** and refactor the GEMM1
  epilogue into a constexpr-callable hook (the ~60-90 LoC the reconstruction scoped): add
  `activation_fn: cutlass.Constexpr` (+ `act_params` tuple) to the ctor, the device `wrapper`
  entry, and the entrypoint; replace the hardcoded `up*silu(gate)` if/else with
  `self.activation_fn(tCompute, acc_vec_gate, acc_vec_up, alpha_val, *act_params)`, defaulting to a
  `swiglu_silu_default` that preserves today's packed+scalar split. Pin the flashinfer source
  version (checklist #4). This is **validator infrastructure**, shipped+tuned once, not per-bundle.
- **`codec/nvfp4.py`** — re-home clean (~250 LoC pure torch, all CPU-validatable): a reference
  NVFP4 quantizer (LUT + per-16 UE4M3 block scale + fp32 global), `swizzle_blockscale` (lift
  sglang's pure-torch `utils.py:597`), `convert_sf_to_mma_layout` (lift flashinfer
  `cute_dsl/utils.py:339`), `interleave_w13_halves` (granularity-64), and the scalar-alpha algebra
  (`w1_alpha[e] = w13_weight_scale_2[e]/used_input_scale`). **Drop** sglang's EP-slicing /
  modelopt shape-coercion glue — re-derive against our own layer view.
- **`override.py`** —
  ```python
  @dataclass(frozen=True)
  class EpiloguePoint:
      base_kernel: str
      compose: Callable        # (override_device_fn) -> a fused_experts(x, ids, w, prepared, out)
      prepare_from_layer: Callable   # validator-owned NVFP4 weight-prep -> `prepared`
      reference: Callable      # fp32 HP reference from dequantized weights (activation-parametric)
  OVERRIDE_POINTS = {"moe.fused_experts/gemm1_epilogue": EpiloguePoint(...)}
  ```
  `compose(override_fn)` JIT-binds the base kernel with the miner's epilogue as the constexpr param
  and returns a callable with the standard `fused_experts(x, topk_ids, topk_weights, prepared, out)`
  signature.

### M1.3 — Load-time composition (the registration path)

At the site where a `KernelImpl` is built from a manifest op (bootstrap `optima/seam.py`
registration **and** the CLI `evaluate`/`bench` registration in `optima/cli.py` — confirm both
sites), branch on `op.base_kernel`:
```python
if op.base_kernel:
    pt = optima_kernels.override.OVERRIDE_POINTS[f"{op.slot}/{op.override_point}"]
    override_fn = load_entry(source, op.entry)        # scanned miner device source
    entry   = pt.compose(override_fn)                 # standard fused_experts signature
    prepare = pt.prepare_from_layer                    # validator-owned, NOT the miner's
    impl = KernelImpl(slot=op.slot, bundle_id=..., entry=entry, prepare=prepare,
                      eligibility=eligibility_from_metadata(meta, op.dtypes))
```
From `dispatch.py`'s view this is an ordinary `moe.fused_experts` kernel: it flows through
`_run_moe_kernel` (`:483`), gets validator output allocation (`:497`), the quant pairing (`:287`),
the `graph_safe` gate (`:282`), and the TP all-reduce replay (`:508`). **No dispatcher or seam
change beyond M0.** A bad override → JIT compile error at `compose()` → registration fails (bundle
rejected), or a fidelity-gate rejection — never a corrupt validator.

### M1.4 — Per-model activation reference + cosine (bring the M3-branch work to main, clean)

The generic `_moe_reference` (`slots.py:392`) computes plain SiLU → it scores the swigluoai kernel
0.0. The activation is a **model fact**, validator-owned. Add to `slots.py`:
```python
@dataclass(frozen=True)
class Activation:           # validator-owned, per model
    kind: str               # "silu" | "swigluoai"
    alpha: float = 0.0      # swiglu sigmoid gain (1.702 for M3) — SEPARATE from the dequant scale
    limit: float = 0.0      # clamp (7.0 for M3)
MODEL_PROFILES: dict[str, dict[str, ...]] = {
    "MiniMax-M3": {"moe.fused_experts": SlotProfile(
        activation=Activation("swigluoai", 1.702, 7.0),
        correctness=Correctness("cosine", min_cosine=0.985))},
}
```
- Make `_moe_reference` activation-aware (`_gated_activation(gate, up, act)`; default SiLU → all
  existing tests + the example bundle unchanged).
- `slot_for_model(name, model_key)` = `get_slot` + `dataclasses.replace` rebinding
  `invoke_reference` to the model's activation and swapping `correctness`→cosine. No model_key →
  identical to `get_slot` (generic path unchanged).
- Thread `model_key` through `verify_entry_from_source` (`verify.py:214`) and the dispatch
  reference. The validator supplies it from the **served model**, never from bundle metadata (a
  miner names the model; the numbers are the validator's). **The two-alpha trap:** `Activation.alpha`
  (1.702, inside the sigmoid) is distinct from the per-expert NVFP4 dequant scale — pass separately.

### M1.5 — The swigluoai submission, re-expressed (the n=1 → n=1-clean proof)

```toml
# manifest.toml
[[ops]]
slot           = "moe.fused_experts"
base_kernel    = "nvfp4_moe_megakernel"
override_point = "gemm1_epilogue"
entry          = "gemm1_epilogue"
source         = "kernels/swigluoai.py"
metadata       = "metadata/moe.json"
```
`kernels/swigluoai.py` = the existing `swigluoai_epilogue_unpacked` (`experiments/minimax_m3/
kernels/swigluoai_epilogue_cutedsl.py:67`) as the `@cute.jit gemm1_epilogue` + the `swigluoai_torch`
reference. No sglang import, no `open()`, no vendored tree, no `prepare`. `metadata/moe.json` keeps
`graph_safe:true`, `quant:["nvfp4"]`, the cosine floor, and the swiglu constexprs.

**Acceptance (M1):** `optima verify --model MiniMax-M3` on the override bundle PASSes on GPU (cosine
≥0.985 via the composed base+override; the generic `--model __generic__` still FAILs as the
swigluoai-vs-SiLU control); the dense CPU path passes; the composed kernel fires through the
`forward_impl` seam under piecewise capture; bracketed B,C,B' reproduces ~1.1× at GSM8K parity.

---

## M2 — Package boundary + recursive scan + blessed-base lockfile

**Why.** Make the moat portable and the measurement consensus-safe.

1. **`optima_kernels` boundary.** `optima/` (harness) imports `optima_kernels`; **never the
   reverse**. `optima_kernels` has zero `sglang` imports — it's the library that dogfeeds the own
   engine. Add a CI guard (`grep -rE "^\s*(import sglang|from sglang)" optima_kernels/` must be empty).
2. **Recursive scan** (`optima/sandbox.py`) — `scan_tree(root)` applies the existing safety denylist
   to **all bundle `.py`**, not just the declared entry (`seam.py:106`, `cli.py` scan only the entry
   today — the vendored-tree hole: a vendored lib using `open()`/`importlib`/`subprocess` is never
   scanned). Wire it into `cmd_scan` / `cmd_verify` / the seam load. This is the
   safety fix (close the hole), NOT a transferability gate — and it does **not** flip to an
   allowlist: per the corrected Axiom 5, a kernel is a kernel (an example legitimately does
   `from sglang.srt...import RMSNorm`), so we don't ban namespaces. Transferability is enforced by
   the contribution *unit* (device-source-at-a-slot/override-point) and, ultimately, the swappable
   arena runtime — not by import hygiene.
3. **Blessed-base lockfile** — `optima/blessed_base.py` (stdlib-only, like `seams.py`): pinned
   versions of `{torch, triton, flashinfer, cutlass/cute-dsl, sgl_kernel, deepgemm}` (+ CUDA/arch).
   Extend `optima compat` (`compat.py`) to assert the resolved versions match — today it checks only
   sglang signatures, but **two validators on different flashinfer pick different kernels →
   different throughput AND numerics → divergent weights → Yuma penalty.** This closes a latent
   consensus bug on the exact surface the win lives on. (Per-arena: fold into `arenas.py` when that
   merges — the `docker_image` should expose the *enumerated, hashed* dep set, not an opaque blob.)

**Acceptance:** the recursive scan rejects a bundle carrying a foreign `.py` that trips the denylist;
`optima compat` reports a flashinfer/cutlass version mismatch; the override bundle still scans clean.

---

## M3 — The second win: `attention.msa_block_score` + `topk_overlap`

Proves the catalog ingests a *structurally different* win (the fp8 MSA indexer — a sub-op whose
output is a **selection set**, not a tensor). The reusable triple: **finer seam + set-metric +
validator owns the irreducible downstream step.**

### M3.1 — `topk_overlap` correctness mode

`optima/slots.py` `Correctness` (`:63`): add
```python
top_k: int = 0           # topk_overlap mode: K
min_overlap: float = 0.0 # topk_overlap mode: required mean |topk(actual) ∩ topk(expected)| / K
```
`optima/verify.py` `_compare` (`:71`): add a branch
```python
if mode == "topk_overlap":
    ta = actual.topk(correctness.top_k, dim=-1).indices
    te = expected.topk(correctness.top_k, dim=-1).indices
    overlap = (ta.unsqueeze(-1) == te.unsqueeze(-2)).any(-1).float().mean(-1)  # per-row
    score = float(overlap.mean())
    return score >= correctness.min_overlap, ..., score, ..., "overlap"
```
`format_verify` (`verify.py:250`): handle the `"overlap"` metric label like `"cosine"`. The output
is the **block scores** (a tensor); the metric compares the *derived top-k sets*, because the
selection is what the model consumes — element-wise score error is irrelevant.

### M3.2 — `attention.msa_block_score` slot

New `SlotSpec` in `slots.py` (kind `"block"`): `make_inputs` (q, index-K, seq_lens, block params),
`out_shapes` → block scores `(B, n_blocks)`, `invoke_reference` = the **bf16** block-score
computation (HP reference), `correctness=Correctness("topk_overlap", top_k=16, min_overlap=0.9375)`
(15/16). Register in `SLOTS` (`:590`). **Invariant note:** the kernel emits *scores only*; the
validator owns the top-k selection AND the downstream bf16 attend — so the kernel stays upstream of
the sampler (nothing to substitute), and a wrong score only mis-selects blocks, caught by the
overlap gate + the e2e KL.

### M3.3 — The MSA-indexer seam (model-specific)

New `SeamAdapter` row in `seams.py` + `integrations/sglang_msa.py` patching the MSA backend's score
kernel (`MiniMaxSparseAttnBackend` / `_decode_score_kernel`). Finer than `attention.decode` (which
is the whole attention). The validator keeps top-k + attend; the miner fills the scores buffer. This
is the most involved milestone (a model-specific chokepoint) and is the *template* for any
selection/intermediate win. (Honest caveat: the fp8 indexer's measured e2e win was ~null at
sustainable concurrency — M3 proves the *mechanism*, not that this kernel ships; the admission
kill-gate below would have flagged it.)

### M3.4 — Served-reference-precision (deferrable; for quant-codec wins)

For a *new quant* (beating NVFP4) the HP reference must be **cleaner than the contested precision**.
Today only `--dtype {bf16,fp16,fp32}` (activation dtype) exists — no served-weight-precision knob.
Add an eval control to serve a clean (bf16/fp8) reference-weights path decoupled from the candidate's
quant. Not needed for swigluoai (its cosine gate uses an fp32 reference from the same
NVFP4-dequantized weights), so defer until a codec submission appears.

**Acceptance (M3):** the fp8 MSA-indexer kernel registers at `attention.msa_block_score`, `optima
verify` gates it on `topk_overlap ≥ 15/16` vs the bf16 reference, and the validator owns
selection+attend. Two structurally different wins (swigluoai epilogue, fp8 indexer) are both
ingestable end-to-end.

---

## Cross-cutting — the admission kill-gate

Before the subnet pays to score a kernel, a cheap pre-check (methodology, lightly enforced): the win
must plausibly move the **e2e serving wall**, not a microbench µs. Require each submission's metadata
to carry a `win_measured` regime tag `(model, ctx, concurrency, GPU, format) → claimed speedup`, and
run a **noop-ceiling** check (skip the op, measure the e2e delta = its max possible win) before the
full bracketed eval. If the op's wall-fraction is below the noise floor, reject early. This is the
discipline that killed the thin-N GEMM (vendor floor) and would have flagged the fp8 indexer
(kernel-time-sum ≠ Amdahl wall fraction) — encode it so the subnet never rewards a rabbit hole.

---

## Sequencing summary

```
M0 (forward_impl seam)            ── prerequisite; cheap; do first
   └─ M1 (override-point + optima_kernels base + per-model ref + swigluoai)   ── the keystone
        ├─ M2 (package boundary + lockfile + allowlist)   ── hardens portability + consensus
        └─ M3 (msa_block_score + topk_overlap + msa seam)  ── proves a 2nd, different win
   (admission kill-gate: fold into the eval entry alongside M1)
```

M0+M1 deliver the first **clean, portable, scorable** win (swigluoai as a 14-line override). M2 makes
it consensus-safe and the library standalone. M3 proves the mechanism generalizes past the MoE family.
At that point the subnet ingests more than one sample — the kickstart condition.
