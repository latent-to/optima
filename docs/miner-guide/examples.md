# Example bundles

The repository examples are development fixtures for specific parts of the
component ABI. They are not a catalog of active crowns, and their comments or
metadata are not validator decisions.

Examples that omit an explicit contribution target rely on legacy singleton
resolution. Before adapting one for submission, add the appropriate
`[competition]` table and revalidate it against the registered target catalog.

## Positive learning examples

| Example | What it demonstrates | Limits |
|---|---|---|
| [`miner_silu_torch`](https://github.com/latent-to/cacheon/tree/main/examples/miner_silu_torch) | smallest CPU-importable `entry(x, out)` bundle | correctness/packaging only; not expected to beat a tuned incumbent |
| [`miner_silu_triton`](https://github.com/latent-to/cacheon/tree/main/examples/miner_silu_triton) | Triton activation implementation and architecture constraints | needs matching GPU/Triton environment |
| [`miner_rmsnorm_triton`](https://github.com/latent-to/cacheon/tree/main/examples/miner_rmsnorm_triton) | pure RMSNorm output ownership in Triton | local example, not crown evidence |
| [`miner_attention_torch`](https://github.com/latent-to/cacheon/tree/main/examples/miner_attention_torch) | `attention.sdpa` block ABI and GQA/MQA reference shape | slow Torch diagnostic implementation |
| [`miner_attention_decode_torch`](https://github.com/latent-to/cacheon/tree/main/examples/miner_attention_decode_torch) | `attention.decode` ABI with `seq_lens` | slow eager diagnostic implementation |
| [`miner_moe_fused_experts_torch`](https://github.com/latent-to/cacheon/tree/main/examples/miner_moe_fused_experts_torch) | MoE `prepare` plus serving `entry` | dense reference-style code, not a quantized fast path |
| [`miner_moe_fused_experts_reduce_torch`](https://github.com/latent-to/cacheon/tree/main/examples/miner_moe_fused_experts_reduce_torch) | distributed MoE entry that owns its trailing reduction | needs multi-rank verification for the real contract |
| [`miner_allreduce_torch`](https://github.com/latent-to/cacheon/tree/main/examples/miner_allreduce_torch) | simplest collective ABI using the supplied group | correctness example, not a competitive collective algorithm |

Start with `miner_silu_torch` for the workflow in
[Your first component bundle](your-first-kernel.md). For a new target, use the
signature in [Kernel ABI](kernel-abi.md), not a superficially similar example.

## Negative and adversarial examples

These bundles are meant to fail or expose a gate:

| Example | Intended lesson |
|---|---|
| [`miner_silu_broken_torch`](https://github.com/latent-to/cacheon/tree/main/examples/miner_silu_broken_torch) | wrong activation math fails CPU correctness |
| [`miner_silu_broken`](https://github.com/latent-to/cacheon/tree/main/examples/miner_silu_broken) | a GPU implementation cannot win by skipping required work |
| [`miner_silu_sparse`](https://github.com/latent-to/cacheon/tree/main/examples/miner_silu_sparse) | sparse corruption can evade naive averages, so tail/disagreement and end-to-end gates matter |
| [`miner_rmsnorm_broken`](https://github.com/latent-to/cacheon/tree/main/examples/miner_rmsnorm_broken) | an incorrect normalization is rejected despite plausible output shape |
| [`miner_setup_demo`](https://github.com/latent-to/cacheon/tree/main/examples/miner_setup_demo) | legacy engine-wide `setup` surface for isolation tests |

`miner_setup_demo` is not a registered component template. No registered
target permits `setup`; a cross-cutting engine change belongs in
[Discovery](discovery-lane.md).

## Override example

[`miner_m3_swigluoai_override`](https://github.com/latent-to/cacheon/tree/main/examples/miner_m3_swigluoai_override)
demonstrates a Torch reference paired with a CuTe-DSL epilogue declaration for
`moe.fused_experts/gemm1_epilogue`.

This example exercises composition and the CPU/dense reference path. The
validator-owned GPU base kernel raises `NotImplementedError`, so the example is
not eligible for GPU qualification through that override point. Read
[Override points](override-points.md) before using it. Provenance fields in its
metadata are descriptive inputs, not evidence produced by this bundle.

## Identity fixtures are not kernels

Two test fixtures are useful for inspecting modern manifest structure:

- [`stack_msa_singleton`](https://github.com/latent-to/cacheon/tree/main/tests/fixtures/stack_msa_singleton)
  shows explicit singleton competition identity and capability metadata;
- [`stack_fused_epilogue_atomic`](https://github.com/latent-to/cacheon/tree/main/tests/fixtures/stack_fused_epilogue_atomic)
  shows explicit atomic identity, member rows, declared CUDA source, dependency
  patch, and reviewed rebuild steps.

They test intake, identity, and publication machinery. Their callable/native
bodies are deliberately minimal and may not implement the live slot ABI. Do not
copy them as performance kernels.

## A safe way to reuse an example

1. Copy only a committed source example whose ABI matches your target.
2. Change `bundle_id` and add explicit `[competition]` identity.
3. Replace descriptive metadata with an honest capability domain.
4. Remove files and declarations your implementation does not use.
5. Run `scan` and `verify` on the matching target environment.
6. Inspect the packaged archive before hosting it.

Do not copy caches, generated binaries, local result directories, machine paths,
wallet material, or performance claims into a proposal. The submission must be
self-contained source and declarations that the validator can reproduce.
