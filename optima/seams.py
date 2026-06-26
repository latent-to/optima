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
    chokepoint: str  # "Class.method" patched — for the compat canary + docs
    slots: tuple[str, ...]  # the slot(s) this adapter serves (cross-ref into slots.py)


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
)

# The modules whose import should trigger seam installation (consumed by bootstrap).
TARGET_MODULES = frozenset(a.target_module for a in SEAM_ADAPTERS)
