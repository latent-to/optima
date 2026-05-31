# Staying current with sglang

"Stay up to date with sglang" is **two** problems, because sglang is both:

- our **baseline** — we score a kernel by its speedup vs sglang's own kernels, so a
  stale baseline means miners optimize against an old frontier (and "wins" may
  already be upstream); and
- our **runtime** — we patch sglang internals (the `SiluAndMul` / `RMSNorm` seams,
  `MultiPlatformOp`, the Engine logprob API, specific `ServerArgs` kwargs), so any
  upgrade can break us.

## The hard constraint: a pinned version (consensus)

You **cannot** have validators on different sglang versions. The exact mechanism,
because it's *why* the pin is non-negotiable rather than a convenience:

Multiple validators independently score the **same** miner and each submit a weight
vector; Bittensor's Yuma consensus penalizes a validator whose weights diverge from
the stake-weighted median. If validator A runs sglang X and validator B runs sglang
Y, they measure different throughput and different KL for the **identical** kernel →
they submit different weights → divergence → validators lose trust/emission and the
subnet's scoring becomes incoherent. So **identical measurement across validators is
a protocol requirement**, and that forces a single pinned version. "Let each miner
pick their base," and even "let each validator pick," are both ruled out by this —
it isn't a fairness nicety, it's consensus.

The sglang version is therefore a **coordinated, pinned subnet parameter**, bumped
deliberately per "season," not "latest on each box." The single source of truth is
`PINNED_SGLANG` in [../optima/compat.py](../optima/compat.py) (currently
`0.5.12.post1`, CUDA 13 — validated end-to-end on an H100: all seams intact, the
broken-bundle gate FAILs, faithful kernels PASS).

## What's actually coupled: the score, not the kernel

The pin sounds like it locks miners to one engine version. It mostly doesn't, and
the distinction governs what miners can and can't do:

- **A miner's submission is a kernel** — Triton/CuteDSL math over raw tensors (e.g.
  `silu(x, out)`). It almost never touches sglang's internals, so the **artifact is
  largely version-agnostic**: the same kernel runs identically on 0.5.9 or 0.6.x
  because it operates on tensors, not on sglang APIs.
- **What's pinned to the version is the *evaluation*, not the artifact:**
  1. the **baseline** the kernel must beat (sglang's own kernel for that op), and
  2. the **invocation context** — the shapes / dtypes / strides the op is called
     with, and whether the op is even reached (fusion can route around it).

So a version bump **re-measures** a kernel's score (its baseline moved, its rank may
change) but usually does **not** invalidate the artifact — most kernels still load
and run. Only a kernel whose win depended on version-specific behavior, or whose op
ABI actually changed, needs a rewrite.

### The insulation layer

Miners code to **Optima's slot ABI** (`entry(x, out)`), not to
`SiluAndMul.forward_cuda`. When sglang changes *how* it invokes an op, Optima's
dispatcher and seam absorb the change — every row in the "what usually breaks" table
below is fixed in Optima code, not in miner code. So the coupling a miner *feels* is
to the slow-moving **slot spec**, not the fast-moving engine. This holds as long as
the op's mathematical contract is stable (silu, rmsnorm, attention I/O); it breaks
only when sglang changes what the op fundamentally *is* (e.g. fuses it away) — which
is a real product change, so re-competing on it is correct.

## What the pin means for miners (read this before worrying)

There are three distinct environments; only two are coupled:

1. **The miner's dev box** — anything they want: any sglang (incl. experimental
   `main`), any GPU. Used to write and locally test the kernel.
2. **The validator's scoring box** — pinned sglang X (eventually the canonical
   8×B200). Produces the score.
3. **The product / managed service** — also pinned sglang X.

(2) and (3) are deliberately the **same stack** — you sell what you score. (1) is
free. So "miners must run the validator's version" is true for **submission** and
mostly false for **development**: develop on whatever you like; just submit a kernel
that wins under the pin.

**"But I need a fix that's only in experimental sglang."** Split by *what* you need
it for:

| You need newer sglang for… | Outcome |
|---|---|
| **dev convenience** (faster iteration, or a working baseline on your hardware) | Fine — only submission is pinned, not your dev box. |
| **a runtime dependency of the win** (speedup only appears under that version's behavior) | Correctly excluded — that win doesn't exist for the shipped product. |
| **a fix that *opens a surface*** (an op that's broken in the pin starts running) | A **bump trigger**, not a permanent block — flag it; it argues for the next bump. |
| **a fix that *is* the optimization** (upstream now does the fast thing) | That headroom is simply gone — fine; the product got faster regardless of who did it. |

**Broken-in-the-pin is headroom, not a blocker.** Concrete: stock sglang's
Blackwell (`sm_120a`) MoE is broken / slow-fallback in 0.5.9. That's not a wall —
it's exactly where a miner can win, by beating the fallback, measured entirely under
the pin. The dev-pod FlashInfer experiment confirms this: correcting MXFP4 scale
layout and disabling PDL makes GPT-OSS-120B TP=4 coherent and faster than the plain
Triton fallback on the RTX PRO 6000 Blackwell box. If a later sglang ships a fast
sm_120 MoE, the validator bumps to it, the headroom shrinks, and the competition
moves to a stronger baseline. Either way a miner never needs to *submit* against
experimental sglang.

The same target was also replicated on a latest-stable dev venv
(`torch==2.12.0+cu130`, `sglang==0.5.12.post1`, `flashinfer-python==0.6.12`).
That newer stack needed a smaller SGLang-side patch: add the SM120
FlashInfer-MXFP4 branch and allocate GPT-OSS TP=4 MXFP4 shards using checkpoint
block-ceil padding (`720` logical intermediate values, `736` loaded values,
`768` CUTLASS-aligned values). On the 4x RTX PRO 6000 Blackwell dev pod, batch-32
GPT-OSS-120B TP=4 decode with piecewise CUDA graph disabled passed the
standardized runner: patched `flashinfer_mxfp4` measured `915.6 tok/s` median
versus plain `triton` at `742.6 tok/s` median, a `1.23x` speedup. The same
runner recorded loader smoke, synthetic SwiGLU cosine `0.9999302`, stock
FlashInfer non-completion, startup time separately from throughput, and one
CUDA-graph-enabled serving sanity run. A previous manual batch-32 run measured
`926.7 tok/s` versus `741.6 tok/s`.

Important baseline correction: the no-graph `742.6 tok/s` Triton run is an
isolation baseline, not the strongest stock SGLang can do. A later best-stock
sweep enabled CUDA graph capture, radix cache, explicit attention-backend
choices, and custom all-reduce finalist checks. Under that stronger setup, stock
SGLang's best successful GPT-OSS-120B TP=4 batch-32 result on the same Blackwell
pod was `830.7 tok/s` (`triton` MoE, `triton` attention, CUDA graph, radix
cache). The patched `flashinfer_mxfp4` path reached `1062.4 tok/s` with the
symmetric CUDA-graph/radix setup plus custom all-reduce, a `1.28x` speedup over
the stronger stock baseline. Stock `attention_backend=flashinfer` did not
complete on this stack, FA3 was unavailable in `sgl_kernel`, and custom
all-reduce did not beat the non-custom stock finalist on the PCIe topology.
A separate long-context sanity run padded prompts to about 2k tokens and forced
1024 generated tokens per prompt at batch 4. Stock measured `321.6 tok/s`, the
patched path measured `402.1 tok/s` (`1.25x`), and both produced the expected
fixed arithmetic answers in all timed iterations.

Trying to respect SGLang's own dependency metadata is still the right hygiene
goal. With `uv --torch-backend cu130 --prerelease=allow`, `sglang==0.5.12.post1`
resolves to `torch==2.11.0+cu130`, `flashinfer-python==0.6.11.post1`,
`sglang-kernel==0.4.2.post2`, and `xgrammar==0.2.0`. That packaged stack avoids
the Torch 2.12 `sglang-kernel` ABI rebuild, and the candidate patch passes both
the loader smoke and the synthetic SwiGLU probe. The full candidate model run
currently aborts in FlashInfer/CUTE RMSNorm with `Expected an MLIR object`, so
the packaged-stack blocker has moved out of MoE layout and into FlashInfer/CUTE
runtime compatibility. That's the current best proof that "broken in the base"
can be converted into a miner-grade throughput win rather than just a
compatibility fix, while also showing why exact dependency capture matters.

**The pin protects miners, too.** Without it, a carefully-tuned kernel could
silently go slower or break the moment a validator auto-updated sglang
mid-competition — a rug-pull through no fault of the miner. The pin is a **stable
target and a stable baseline**; because bumps are announced and re-baselined (below),
version changes are scheduled events, not surprises.

## Why bump, and how often: the cadence trade-off

The mission is to push the frontier, so the pin can't sit still — a stale base means
miners optimize things already fixed upstream, and the product falls behind what it
claims to be ("SOTA inference stack"). But bumping isn't free, so treat each bump as
a **coordinated protocol event — structurally a hard-fork:** pin for consensus, then
cut the whole validator set over to the new version at an agreed block, with a
re-baseline (next section).

Cadence is a dial with two failure modes:

- **Too slow** → stale base; wins that already exist upstream; good miners leave.
- **Too fast** → champions re-ranked constantly; miner ROI uncertain; tuning wasted.

Rule of thumb: a champion should earn over a **meaningful window** before a bump can
unseat it. So bump on a deliberate cadence (e.g. monthly, or per sglang minor)
**or** on a trigger (a release that materially improves the product or opens an
important surface) — whichever comes first — never on every upstream commit. The
decision weighs the new version's net effect: product quality + surfaces opened −
surfaces closed − re-baseline churn.

## The bump process (safe + coordinated)

1. **Watch releases.** The clone at `optima/sglang` has the upstream remote;
   `git -C sglang fetch origin --tags` surfaces new tags. (Or watch GitHub releases
   for sgl-project/sglang.)
2. **Static canary.** In a scratch venv, `uv pip install sglang==<new>`, then
   `optima compat`. It introspects the installed sglang (imports + signatures, no
   GPU) and asserts every seam/API we depend on still exists.
3. **Behavioral smoke (on the pod).** If the canary is green, confirm the seam
   still *fires*: `optima bench <broken-bundle>` must still **FAIL** the gate and a
   faithful bundle must behave. A green canary is necessary but not sufficient.
4. **Coordinate + re-baseline.** If both pass: update `PINNED_SGLANG`, announce a
   bump at a block height so **all validators upgrade together**, and
   **re-baseline the champion** — re-score the reigning champion against the *new*
   sglang baseline (the baseline moved, exactly like Affine refreshing its task
   pool; a champion's old speedup isn't comparable to challengers scored on the new
   sglang).
5. **If the canary is RED:** write a small adapter in `optima/integrations/` +
   `optima/seam.py` (the seams are deliberately tiny and isolated for this), then
   re-run from step 3.

## What usually breaks, and where to fix it

| sglang change | canary catches it as | fix in |
|---|---|---|
| seam class renamed/moved (`SiluAndMul`, `RMSNorm`, `MultiPlatformOp`) | `seam: …` FAIL | the import in `integrations/*` + `bootstrap._TARGETS` |
| `forward_cuda` signature change (e.g. residual handling) | `seam: …` detail shows new params | `dispatch.py` dispatcher |
| Engine / `ServerArgs` API change | `Engine.generate …` / `ServerArgs …` FAIL | `eval/_launch.py`, `EvalConfig` |
| a real plugin framework lands (bleeding-edge sglang has one) | (canary still green) | optionally swap the `.pth` for the entry-point plugin — `integrations/sglang_plugin.py` already exists for that |
| compile/graph path imports the swapped kernel by name (0.5.12+ piecewise CUDA graph / torch.compile) | (canary green; the *candidate* launch crashes `ModuleNotFoundError: optima_kernel_*`) | `sandbox.load_entry` registers the kernel module in `sys.modules` before exec |
| seam install races a partially-initialized sglang module on import | (canary green; a caught `optima: failed to install a seam` traceback on every `import sglang`) | `integrations/*` install() guards on the class attribute, not the raising import |

## Who decides the pin (governance & centralization risk)

Choosing the pinned version is a **trust point**: the operator could, in principle,
pin a version that favors a particular miner, or be slow to bump and let the base
rot. Name it rather than pretend it away.

- **Now (single operator):** mitigate with **transparency** — the pinned version is
  public and committed on-chain (it's a subnet parameter, not a private setting), so
  anyone can audit what is being scored against.
- **Mature state:** move the pin under **validator-voted governance** (validators
  signal the next version; bump when a stake threshold agrees), so no single party
  controls the substrate the whole competition runs on.

This belongs in the production blueprint, not today's harness — documented here so it
isn't a surprise later.

## Strategic: upstream or moat?

Decide per winning kernel whether it goes **upstream** to sglang (frontier mission;
the baseline rises and the subnet must keep finding new wins) or stays **private**
(a proprietary stack — the managed-service moat). Likely: the subnet's *composed
stack* is the product; you track sglang as the moving base and your stack sits on
top.

## Automation (set up)

Turn "stay current" into a notification instead of a chore. Two pieces ship in the
repo:

- **`scripts/check_sglang.py`** — checks PyPI for a newer sglang vs `PINNED_SGLANG`
  (pure HTTP, runs anywhere) and, if sglang is importable, runs the seam canary.
  Exit 1 = attention needed. Run it anywhere: `python scripts/check_sglang.py`.
- **`.github/workflows/sglang-canary.yml`** — a weekly GitHub Action (Mondays
  09:00 UTC) that runs the script. **Activates once this repo is on GitHub**; a
  failing run (new release and/or red canary) shows on the Actions tab and emails
  the owner. This is the home for the automation — and it fits the plan to make the
  repo public.

Until the repo is on GitHub, run it locally (e.g. cron on an always-on box):

```
0 9 * * 1  cd /path/to/optima && .venv/bin/python scripts/check_sglang.py
```

Note: the *seam* canary needs sglang importable (best on a GPU/pod venv); the
*release* check works everywhere. On a CPU CI runner sglang may not build, so the
Action reliably catches new releases and the full seam check runs on the pod as
part of the bump process.
