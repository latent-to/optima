# Dependency patches

A dependency patch is a narrowly allowed source delta against a pinned
validator dependency. It exists for registered targets whose callable boundary
depends on a small producer/export change in that dependency.

It is not a framework-mode escape hatch, a way to edit `site-packages`, or a way
to execute a bundle-supplied installer.

## Registered dependency-patch policy

The dependency-patch allowlist admits only `flashinfer` files matching:

```text
flashinfer/data/csrc/fused_moe/*
```

The target catalog grants the FlashInfer patch features only to the deep
`collective.moe_finalize_ar_rmsnorm` singleton target and the
`collective.moe_epilogue.v1` atomic target. A patch on another registered target
does not gain permission merely because the manifest parses.

The authoritative allowlist is in
[dep_policy.py](https://github.com/latent-to/cacheon/blob/main/optima/dep_policy.py),
and target feature policy is in
[target_catalog.py](https://github.com/latent-to/cacheon/blob/main/optima/target_catalog.py).

## Declaration

Add the text diff to the bundle and declare it:

```toml
[[dep_patches]]
target = "flashinfer"
path = "patches/fused_moe_export.patch"
```

The patch must be UTF-8 unified diff text using canonical dependency-relative
paths, for example:

```diff
--- a/flashinfer/data/csrc/fused_moe/example.cu
+++ b/flashinfer/data/csrc/fused_moe/example.cu
@@ -10,1 +10,1 @@
-old_call();
+new_call();
```

Only modifications and new text files are supported. The parser rejects
deletions, renames, copies, binary patches, unsafe paths, duplicate file
sections, malformed hunks, and missing-newline markers. Application is byte
exact at the declared location: there is no fuzzy context match or offset
search. A patch prepared against a different dependency revision must fail.

See [deppatch.py](https://github.com/latent-to/cacheon/blob/main/optima/deppatch.py).

## Rebuild plan

A declared patch also needs the validator-owned patch applier selected in
`rebuild.json`:

```json
{
  "steps": [
    {"type": "repo_python", "path": "apply_dep_patch.py"}
  ]
}
```

If the bundle also declares standalone CUDA extension sources, select the
approved extension builder too:

```json
{
  "steps": [
    {"type": "repo_python", "path": "apply_dep_patch.py"},
    {"type": "repo_python", "path": "build_cuda_ext.py"}
  ]
}
```

Those are the registered dependency-patch rebuild patchers. The same closed registry also
contains `build_cute_cubin.py` (`optima.build-cute-cubin.v1`) for the sealed
direct-artifact provider; it is not a dependency-patch installer. The parser rejects
`bundle_python`, arbitrary repository scripts, duplicate patchers, unknown
fields, and unregistered step types. The validator snapshots each patcher's ID
and source digest; the bundle chooses a reviewed capability, not executable
installer code.

The rebuild contract is implemented in
[rebuild.py](https://github.com/latent-to/cacheon/blob/main/optima/rebuild.py).

## Build and load are separated

Production does not compile inside the serving process:

1. The build phase copies the allowed pinned dependency subtree into a private
   stage, applies the exact patch, inventories the entire overlay, and builds
   only validator-approved native products.
2. The publication is content-addressed and made immutable with its complete
   build identity and provenance.
3. The isolated engine load phase reopens and validates that publication. It
   may load the exact artifact but cannot compile, repair, or fall back to an
   unrecorded JIT product.

The combined `all` phase exists only for direct development diagnostics. Its
local cache is not authoritative evidence.

This split prevents a bundle ID, ambient cache, or mutable shared install from
selecting what runs. The overlay validation logic is in
[dep_policy.py](https://github.com/latent-to/cacheon/blob/main/optima/dep_policy.py),
and the reviewed applier is
[apply_dep_patch.py](https://github.com/latent-to/cacheon/blob/main/optima/patchers/apply_dep_patch.py).

## Native source declarations

If your target-approved implementation includes inspectable native source,
declare every `.cu`/`.cuh` file on the relevant op row:

```toml
[[ops]]
slot = "collective.moe_finalize_ar_rmsnorm"
source = "kernels/finalize.py"
entry = "moe_finalize_ar_rmsnorm"
cuda_sources = ["kernels/finalize_sm103.cu"]
architectures = ["sm103"]
```

The builder includes all declared sources in provenance but compiles only the
variants eligible for the bound architecture. Compiler inputs and dependency
files are checked against the selected source set and pinned build roots. A
prebuilt `.so` in the bundle is never an acceptable substitute.

## Development and review checklist

Before proposing a dependency-patch bundle:

- confirm the requested registered target allows both the dependency patch and
  its rebuild feature;
- generate the diff against the exact pinned FlashInfer source;
- keep the change inside the fused-MoE allowlist;
- declare every native source and architecture;
- run `scan` and distributed `verify` on the matching environment;
- inspect graph replay and end-to-end developer diagnostics;
- review the complete materialized delta, not just the Python shim.

The committed
[atomic stack fixture](https://github.com/latent-to/cacheon/tree/main/tests/fixtures/stack_fused_epilogue_atomic)
demonstrates manifest identity, patch declaration, and rebuild selection. Its
native and Python bodies are deliberately test stubs and do not implement the
production deep ABI. Treat it as a schema fixture, not a miner kernel template.

If the desired edit is outside this closed policy, submit it through the
[Discovery lane](discovery-lane.md) rather than disguising it as a dependency
patch.
