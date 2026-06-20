"""Validator-owned kernel registry + eligibility + active toggle.

The dispatcher (see ``optima/dispatch.py``) consults a single process-global
registry to decide, per call, whether to use a miner kernel or fall back to the
baseline. The registry is owned by the validator harness; the miner never writes
to it. Registration happens *after* the manifest is validated, the source is
scanned, and the entry callable is loaded inside the isolated worker.

The ``active`` flag lets the end-to-end eval flip between the *reference* run
(miner disabled -> baseline kernels) and the *candidate* run (miner enabled) in
a single engine process with identical weights, so KL and speedup isolate the
op's effect exactly.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Eligibility:
    """Declarative gate parsed from a bundle op's metadata.

    A miner kernel is only used when the live tensors match what it claims to
    support. Anything outside the claim falls back to baseline rather than
    risking a wrong or unsafe launch.
    """

    dtypes: frozenset[str] = frozenset()
    architectures: frozenset[str] = frozenset()
    max_last_dim: Optional[int] = None  # cap on x.shape[-1]
    # The miner DECLARES the kernel is CUDA-graph-capturable: static shapes, no host
    # syncs (.item()/.cpu()), no data-dependent Python control flow, writes only to the
    # validator-allocated buffer. Required to run the block/collective seams under the
    # scoring config (graphs ON) — that is the ONLY regime a real MoE/comms win is worth
    # anything in (graphs-off cripples the baseline ~4.5-6.5x). Default False: an
    # undeclared kernel stays eager-only (so it can't wedge graph capture); the seam
    # falls back to the trusted baseline in-graph. A kernel that lies (declares graph_safe
    # but isn't) either errors at capture -> fallback, or is caught by the fidelity gate.
    graph_safe: bool = False
    # The quantization formats this kernel's (prepare, forward) handle, e.g.
    # ``{"nvfp4"}`` or ``{"fp8"}``. EMPTY (default) means the kernel takes DENSE
    # (unquantized) expert weights only. The MoE dispatcher pairs a kernel to a layer
    # by format: a dense layer runs only an empty-``quant`` kernel; a quantized layer
    # runs only a kernel that declares its exact format (else fall back to the trusted
    # baseline). This is the gate that lets an NVFP4 expert kernel reach the seam without
    # feeding a dense kernel packed FP4 bytes + separate scales it would mis-read.
    quant: frozenset[str] = frozenset()

    def accepts(self, *, dtype_name: str, last_dim: int, arch: Optional[str]) -> bool:
        if self.dtypes and dtype_name not in self.dtypes:
            return False
        if self.architectures and arch is not None and arch not in self.architectures:
            return False
        if self.max_last_dim is not None and last_dim > self.max_last_dim:
            return False
        return True


@dataclass
class KernelImpl:
    slot: str
    bundle_id: str
    entry: Callable[..., Any]  # called as entry(*inputs, out)
    # Optional 2nd callable for (prepare, forward) slots (e.g. moe.fused_experts): the
    # validator-owned dispatcher runs it ONCE on the layer's raw weights at the first
    # call and memoizes the result, then passes that `prepared` to `entry` each step.
    # None for plain forward-only slots (silu / rmsnorm / attention).
    prepare: Optional[Callable[..., Any]] = None
    eligibility: Eligibility = field(default_factory=Eligibility)


class KernelRegistry:
    """Process-global registry. One active bundle at a time (MVP)."""

    def __init__(self) -> None:
        self._by_slot: dict[str, KernelImpl] = {}
        self._active: bool = False
        self._strict: bool = False  # if True, a kernel exception aborts instead of falling back
        self._lock = threading.Lock()

    # ---- registration (validator-side) ----

    def register(self, impl: KernelImpl) -> None:
        with self._lock:
            self._by_slot[impl.slot] = impl

    def clear(self) -> None:
        with self._lock:
            self._by_slot.clear()
            self._active = False

    # ---- active toggle (used by the eval to swap reference<->candidate) ----

    def enable(self) -> None:
        self._active = True

    def disable(self) -> None:
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def strict(self) -> bool:
        return self._strict

    def set_strict(self, value: bool) -> None:
        self._strict = value

    # ---- lookup (dispatcher-side, hot path) ----

    def lookup(
        self, slot: str, *, dtype_name: str, last_dim: int, arch: Optional[str]
    ) -> Optional[KernelImpl]:
        if not self._active:
            return None
        impl = self._by_slot.get(slot)
        if impl is None:
            return None
        if not impl.eligibility.accepts(dtype_name=dtype_name, last_dim=last_dim, arch=arch):
            return None
        return impl

    def slots(self) -> list[str]:
        return sorted(self._by_slot)


# A single process-global registry. The dispatcher and the eval both reach this.
REGISTRY = KernelRegistry()


def eligibility_from_metadata(meta: dict | None, manifest_dtypes: tuple[str, ...]) -> Eligibility:
    """Build an Eligibility from a bundle op's metadata json (+ manifest dtypes)."""
    meta = meta or {}
    dtypes = set(manifest_dtypes) | {str(d) for d in meta.get("dtypes", ())}
    archs = {str(a) for a in meta.get("architectures", ())}
    max_last = meta.get("max_last_dim")
    return Eligibility(
        dtypes=frozenset(dtypes),
        architectures=frozenset(archs),
        max_last_dim=int(max_last) if max_last is not None else None,
        graph_safe=bool(meta.get("graph_safe", False)),
        quant=frozenset(str(q) for q in meta.get("quant", ())),
    )
