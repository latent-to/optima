# Override points

Override points let a small miner-owned function fill a typed hole in a
validator-owned base kernel. The resulting callable still implements a normal
registered slot ABI; the base kernel, preparation path, output ownership, and
serving integration remain validator-owned.

This is an advanced composition mechanism, not permission to patch arbitrary
engine code.

## Registered capability

The override registry defines one override-point identity:

```text
moe.fused_experts/gemm1_epilogue
```

It selects the `gemm1_epilogue` hole in the validator-owned
`nvfp4_moe_megakernel` base. The registry and composition code are in
[override.py](https://github.com/latent-to/cacheon/blob/main/optima_kernels/override.py).

Its support boundary is explicit:

- manifest parsing, point lookup, composition, and the dense/CPU reference path
  are implemented and tested;
- the validator-owned GPU `nvfp4_moe_megakernel.run` path raises
  `NotImplementedError`; and
- therefore the committed example supports ABI development but is not eligible
  for GPU qualification through this base.

See the explicit status in
[nvfp4_megakernel.py](https://github.com/latent-to/cacheon/blob/main/optima_kernels/moe/nvfp4_megakernel.py).
Provenance measurements embedded in example metadata are not evidence produced by
the override bundle.

## Bundle declaration

A competitive version must still request its registered singleton target:

```toml
bundle_id = "my-moe-epilogue-v1"
abi_version = "optima-op-abi-v0"

[competition]
target = "moe.fused_experts"
mode = "slot"

[[ops]]
slot = "moe.fused_experts"
source = "kernels/epilogue.py"
entry = "gemm1_epilogue"
base_kernel = "nvfp4_moe_megakernel"
override_point = "gemm1_epilogue"
dtypes = ["bfloat16", "float16"]
metadata = "metadata/moe.json"
```

The target catalog must allow the observed `override` feature, and the point
registry must resolve the `(slot, override_point)` pair. Manifest fields cannot
name a miner-supplied base kernel.

## The paired callable convention

For `entry = "gemm1_epilogue"`, the loader expects:

- `gemm1_epilogue_ref(gate, up, ...)`: a required Torch reference used by the
  dense path and fidelity checking;
- `gemm1_epilogue(...)`: the optional-at-import device function, required for a
  real GPU implementation.

On a CPU host, guard CuTe/CUTLASS imports so the reference remains importable:

```python
def gemm1_epilogue_ref(gate, up):
    return activation(gate, up)

try:
    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def gemm1_epilogue(t_compute, gate, up, alpha_val, *act_params):
        ...
except ImportError:
    pass
```

If the device implementation calls `cute.compile(...)`, static closure accepts
that method only when `cute` is bound exactly once by `import cutlass.cute as
cute` or the equivalent absolute `from cutlass import cute`. Rebinding or
shadowing the alias, importing it twice, using another module's `.compile`,
vendoring a local `cutlass` package, or passing a string/bytes literal as the
first positional argument fails closed. The validator-owned
`DSL_JIT_ENTRYPOINTS` table admits only CuTe. The exception exists only
for CuTe's function-object tracing API; it does not permit string-to-code
compilation or `builtins.compile`. Triton's lazy `@triton.jit` launch path does
not need an explicit `.compile` admission.

The authoritative admission logic is
[`dsl_jit_policy.py`](https://github.com/latent-to/cacheon/blob/main/optima/dsl_jit_policy.py)
and [`engine_tree.py`](https://github.com/latent-to/cacheon/blob/main/optima/engine_tree.py).

The exact device ABI belongs to the registered point. For this point,
`alpha_val` is the per-expert quantization/dequantization value; activation
parameters such as a sigmoid gain are distinct values. Mixing those two
meanings is a correctness bug.

## What composition does

The loader resolves the point, loads the required `_ref` function and optional
device function, and builds a standard `fused_experts` callable. In dense mode,
the validator-owned base implementation evaluates the miner's Torch activation.
In GPU mode, the base would install the device epilogue inside the megakernel.

The miner does not supply `prepare` for this override. The validator owns the
base kernel's weight layout and returns the composed `(entry, prepare)` pair.

## Development checks

Study the committed
[swigluoai override example](https://github.com/latent-to/cacheon/tree/main/examples/miner_m3_swigluoai_override),
then run the source scan and focused composition tests:

```bash
python -m optima.cli scan examples/miner_m3_swigluoai_override
python -m pytest -q tests/test_optima_kernels.py tests/test_model_profiles.py
```

The example declares only BF16/FP16 NVFP4 eligibility. A normal CPU `verify`
descriptor is dense and therefore correctly reports every shape N/A; do not
misread that as a composition pass. The focused tests exercise the dense
reference mechanism explicitly. None of these checks tests the missing GPU base,
graph replay, authoritative quality evidence, or performance.

## When not to use an override

Use a normal slot implementation when you own the complete registered callable.
Use a [dependency patch](dep-patches.md) only when a registered target explicitly
requires an approved dependency export/build change. Use the
[Discovery lane](discovery-lane.md) when the desired hole, base, or engine change
is not registered. Do not vendor a large dependency into a bundle to simulate a
validator-owned base.
