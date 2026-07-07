"""Single source of truth for the seam ADAPTERS (the sglang chokepoints we patch).

Three places must agree on the set of seam adapters: the ``.pth`` bootstrap (which
modules to watch for import), ``seam.activate()`` (which adapters to install), and the
``compat`` canary (which chokepoints to assert survived an sglang bump). Keeping three
hand-maintained lists in lockstep is exactly the drift sglang itself fought — a clean
registry PLUS a parallel hardcoded choices list turned "add one backend" into an
N-file edit (see the project review). So all three derive from the ONE table here:
adding a seam is a single entry, and the bootstrap watch-list, the install loop, and
the canary all pick it up.

This is deliberately SEPARATE from ``slots.py``. ``SlotSpec`` is the miner-facing
contract — frozen and model-agnostic. A seam adapter is version-pinned glue to a
specific sglang internal that churns on every ``PINNED_SGLANG`` bump. Coupling them
would tie the stable contract to sglang's internals, the exact inversion of the LLVM
lesson (freeze the contract, let the adapters churn). The ``slots`` field below is a
cross-REFERENCE (which slots an adapter serves), not the source of either.

Import-light on purpose (stdlib only): the ``.pth`` bootstrap imports this at
interpreter startup, before — and without — importing torch or sglang.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeamAdapter:
    name: str  # short label (also the integration module stem: optima.integrations.sglang_<name-ish>)
    target_module: str  # the sglang module whose import triggers installation
    integration: str  # the optima.integrations submodule exposing install(registry)
    # The patched chokepoint, for the compat canary + docs: "Class.method" for a method
    # patch, a bare "function_name" (no dot) for a module-LEVEL function rebind, or
    # "attr:Name" for a (possibly non-callable) module attribute the adapter rebinds.
    chokepoint: str
    slots: tuple[str, ...]  # the slot(s) this adapter serves (cross-ref into slots.py)
    # Package that must be importable for this adapter's row to be ASSESSABLE. The
    # compat canary SKIPS (not fails) the row when it is absent — e.g. flashinfer
    # exists on engine boxes but not on CPU intake/dev boxes. None = always assessable.
    requires: str | None = None


# THE table. Add a seam here and the bootstrap watch-list, the activate() install loop,
# and the compat canary all pick it up — no parallel list to keep in sync.
SEAM_ADAPTERS: tuple[SeamAdapter, ...] = (
    SeamAdapter("activation", "sglang.srt.layers.activation",
                "sglang_silu", "SiluAndMul.forward_cuda", ("activation.silu_and_mul",)),
    SeamAdapter("layernorm", "sglang.srt.layers.layernorm",
                "sglang_norm", "RMSNorm.forward_cuda", ("norm.rmsnorm",)),
    SeamAdapter("attention", "sglang.srt.layers.radix_attention",
                "sglang_attention", "RadixAttention.forward", ("attention.decode", "attention.sdpa")),
    SeamAdapter("moe", "sglang.srt.layers.moe.fused_moe_triton.layer",
                "sglang_moe", "FusedMoE.forward_impl", ("moe.fused_experts", "moe.fused_experts_reduce")),
    SeamAdapter("collective", "sglang.srt.distributed.parallel_state",
                "sglang_allreduce", "GroupCoordinator.all_reduce", ("collective.all_reduce",)),
    # Module-LEVEL function chokepoint (no dot): sglang's fused AR+residual+RMSNorm
    # epilogue waist. Callers resolve the symbol per call via a function-local import,
    # so rebinding the module attribute reroutes every call site. Only hot when the
    # arena serves --enable-flashinfer-allreduce-fusion (arena server flag).
    SeamAdapter("arfusion", "sglang.srt.layers.flashinfer_comm_fusion",
                "sglang_arfusion", "flashinfer_allreduce_residual_rmsnorm",
                ("collective.ar_residual_rmsnorm",)),
    # The deep fused-epilogue pair (see optima/moe_export.py). defer_gate records the
    # per-layer "this AR is deferred" decision + forward/layer scoping on the
    # model-agnostic LayerCommunicator idiom; moe_export arms skip-finalize around the
    # cutlass fused-moe call and pends the exported pre-finalize pointers. The consume
    # side is the EXISTING arfusion dispatcher (it checks pends before the shallow
    # path). Both install only for bundles that registered the deep slot.
    SeamAdapter("defer_gate", "sglang.srt.layers.communicator",
                "sglang_defer_gate", "LayerCommunicator.should_fuse_mlp_allreduce_with_next_layer",
                ("collective.moe_finalize_ar_rmsnorm",)),
    SeamAdapter("moe_export", "sglang.srt.layers.quantization.modelopt_quant",
                "sglang_moe_export", "flashinfer_cutlass_fused_moe",
                ("collective.moe_finalize_ar_rmsnorm",), requires="flashinfer"),
    # NOT a slot seam: the dep_patches runtime consume side. When the active bundle
    # declared dependency patches (materialized as a csrc OVERLAY by the reviewed
    # patcher — optima/patchers/apply_dep_patch.py), this adapter repoints
    # flashinfer's late-bound csrc constant at the overlay, forces JIT (bypassing the
    # prebuilt AOT .so) for the policy's module names, and writes an `overlay`
    # receipt. Empty slots tuple; policy comes from optima/dep_policy.py, never from
    # bundle content. Watching flashinfer.jit.core (imports .env transitively) means
    # the rebind lands before any JIT gen can run.
    SeamAdapter("flashinfer_overlay", "flashinfer.jit.core",
                "flashinfer_overlay", "attr:JitSpec", (), requires="flashinfer"),
)

# The modules whose import should trigger seam installation (consumed by bootstrap).
TARGET_MODULES = frozenset(a.target_module for a in SEAM_ADAPTERS)
