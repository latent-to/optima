"""The blessed dependency base — the kernel-library surface the subnet scores on.

A miner kernel runs against these libraries and a base override-kernel is composed from them,
so for CONSENSUS they must be identical across validators: two validators on different
flashinfer (or cutlass / triton) JIT *different* kernels -> different throughput AND numerics
-> divergent weight vectors -> Yuma penalty. Today only sglang is pinned
(``compat.PINNED_SGLANG``); the kernel libs ride along implicitly. This makes the whole import
surface an explicit, canary-checked pin, exactly like ``PINNED_SGLANG``.

stdlib-only (no torch import) so the canary runs anywhere — like ``seams.py``. Per-arena: when
``arenas.py`` merges this becomes part of the ``Arena`` (the ``docker_image`` should expose this
enumerated, hashed set, not an opaque blob).

The pinned ``version`` is ``None`` for now = **record-only**: the canary reports the installed
version (the consensus-audit surface) but does not enforce, because the exact arena versions
aren't validated yet. Set a version to enforce it (a mismatch then fails the canary, like the
sglang pin). The ENFORCEMENT mechanism is what M2 ships; the values are finalized per arena.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PinnedDep:
    dist: str  # installed distribution name (importlib.metadata.version)
    version: Optional[str]  # the pinned version, or None = record-only (not yet enforced)
    why: str


# THE blessed base. A kernel is a kernel — these are all kernel libraries (Axiom 5 excludes
# engine orchestration, not kernel libs). Versions finalized per arena; flashinfer is the
# override base (the M3 win used 0.6.12).
BLESSED_BASE: tuple[PinnedDep, ...] = (
    PinnedDep("torch", None, "tensor runtime + the dtype/layout contract at every seam"),
    PinnedDep("triton", None, "Triton kernels (the lingua franca tier)"),
    PinnedDep("flashinfer", None, "fused MoE / attention CuTe-DSL kernels (the override base; win used 0.6.12)"),
    PinnedDep("nvidia-cutlass-dsl", None, "CuTe-DSL (CUTLASS python) — the device-code substrate"),
    PinnedDep("sgl-kernel", None, "sglang's CUDA kernel package (a kernel lib; importable)"),
    PinnedDep("deepgemm", None, "DeepGEMM FP8/FP4 GEMMs (optional kernel lib)"),
)


def resolved_version(dist: str) -> Optional[str]:
    """The installed version of a distribution, or None if not installed."""
    try:
        import importlib.metadata as md

        return md.version(dist)
    except Exception:  # noqa: BLE001 - not installed / no metadata
        return None


def check_blessed_base(base: tuple[PinnedDep, ...] = BLESSED_BASE) -> list[tuple[str, bool, str]]:
    """Per-dep (name, ok, detail) for the compat canary. Record-only deps (version None) are
    always ok and just report the installed version; an enforced dep fails if absent or
    version-mismatched — the consensus break a flashinfer/cutlass skew would silently cause."""
    rows: list[tuple[str, bool, str]] = []
    for dep in base:
        inst = resolved_version(dep.dist)
        if dep.version is None:
            rows.append((dep.dist, True, f"installed={inst} (record-only; pin per arena)"))
        elif inst is None:
            rows.append((dep.dist, False, f"NOT INSTALLED (pinned {dep.version})"))
        elif inst == dep.version:
            rows.append((dep.dist, True, f"installed={inst} == pinned"))
        else:
            rows.append((dep.dist, False, f"installed={inst} pinned={dep.version}  <-- DIFFERS"))
    return rows
