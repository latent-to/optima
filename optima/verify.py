"""Op-correctness — the cheap gate before any end-to-end eval.

Given a slot and a miner ``entry`` callable, generate deterministic inputs over the
slot's standard shapes, run the miner kernel and the trusted *high-precision*
reference, and compare under the slot's ``Correctness`` policy:

* ``allclose`` — every element within ``atol + rtol*|e|`` (numerically-equivalent
  ops, e.g. a faster silu).
* ``matched_ratio`` — at least ``min_ratio`` of elements within that bound (kernels
  that legitimately differ from the reference: attention's reordered softmax, fp8,
  MLA weight absorption). The reference is always high-precision ground truth, never
  the stock kernel — so a faster *and slightly different* kernel can still pass.

Multi-output slots (blocks) are supported: the validator allocates one ``out`` per
declared output shape and the miner fills them.

This is the per-op analogue of a unit test: necessary but NOT sufficient — small
per-op errors that pass here can compound into large end-to-end KL, which is why the
pipeline still runs the end-to-end gate. The seeds/shapes are re-randomizable per
epoch by the caller so a kernel cannot special-case the fixed verification inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch

from optima.slots import SlotSpec


@dataclass
class ShapeResult:
    shape: dict
    dtype: str
    passed: bool
    max_abs_err: float
    max_rel_err: float
    pass_ratio: float = 1.0  # fraction within tol (matched_ratio) OR cosine (cosine mode); informative
    detail: str = ""
    metric: str = "ratio"  # label for pass_ratio: "ratio" | "cosine"


@dataclass
class VerifyResult:
    slot: str
    dtype: str
    passed: bool
    shape_results: list[ShapeResult]

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.shape_results if not r.passed)


def _as_list(x) -> list:
    """Normalize a slot's reference/out_shapes return to a list.

    Accepts a bare tensor or bare shape-tuple (single-output slots may return one
    directly) as well as an explicit sequence (multi-output blocks)."""
    if isinstance(x, (list, tuple)) and (len(x) == 0 or not isinstance(x[0], int)):
        return list(x)
    return [x]


def _compare(
    actual: torch.Tensor, expected: torch.Tensor, *, atol: float, rtol: float, correctness
) -> tuple[bool, float, float, float, str, str]:
    # Returns (passed, max_abs, max_rel, score, detail, metric_label).
    if actual.shape != expected.shape:
        return False, float("inf"), float("inf"), 0.0, f"shape mismatch {tuple(actual.shape)} vs {tuple(expected.shape)}", "ratio"
    a = actual.float()
    e = expected.float()
    if not torch.isfinite(a).all():
        return False, float("inf"), float("inf"), 0.0, "actual has non-finite values", "ratio"
    abs_err = (a - e).abs()
    rel_err = abs_err / (e.abs() + 1e-12)
    mode = correctness.mode
    if mode == "cosine":
        # Low-bit fidelity: direction (and optionally energy) vs the HP reference.
        cos = float(torch.nn.functional.cosine_similarity(a.flatten(), e.flatten(), dim=0))
        ne = float(e.flatten().norm())
        rel_norm = abs(float(a.flatten().norm()) - ne) / (ne + 1e-12)
        ok_cos = cos >= correctness.min_cosine
        ok_norm = correctness.max_rel_norm_err <= 0 or rel_norm <= correctness.max_rel_norm_err
        passed = ok_cos and ok_norm
        if passed:
            detail = ""
        elif not ok_cos:
            detail = f"cosine {cos:.5f} < min_cosine {correctness.min_cosine}"
        else:
            detail = f"rel_norm_err {rel_norm:.3f} > {correctness.max_rel_norm_err}"
        return passed, float(abs_err.max()), float(rel_err.max()), cos, detail, "cosine"
    slack = atol + rtol * e.abs()  # allclose: |a-e| <= atol + rtol*|e|
    within = abs_err <= slack
    ratio = float(within.float().mean())
    if mode == "matched_ratio":
        passed = ratio >= correctness.min_ratio
        detail = "" if passed else f"matched {ratio:.4f} < min_ratio {correctness.min_ratio}"
    else:
        passed = bool(within.all())
        detail = ""
    return passed, float(abs_err.max()), float(rel_err.max()), ratio, detail, "ratio"


def verify_entry(
    slot: SlotSpec,
    entry: Callable[..., None],
    *,
    prepare: Optional[Callable] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
    seed: int = 0,
    shapes: Optional[list[dict]] = None,
) -> VerifyResult:
    """Verify a miner ``entry`` against the slot's reference.

    ``entry`` is called via ``slot.invoke_entry(entry, inputs, outs, prepared)`` and
    must write its result into the validator-allocated tensors in ``outs``. For a
    *(prepare, forward)* slot (``slot.prepare`` set, e.g. ``moe.fused_experts``) pass
    the miner's ``prepare`` callable too — it runs once on the raw weights and its
    result is handed to ``entry`` as ``prepared`` (otherwise ``prepared`` is None).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tol = slot.tolerance_for(dtype)
    test_shapes = shapes if shapes is not None else list(slot.shapes)

    results: list[ShapeResult] = []
    for i, shape in enumerate(test_shapes):
        inputs = slot.make_inputs(dtype=dtype, device=device, seed=seed + i, **shape)
        expected = _as_list(slot.invoke_reference(inputs))
        out_shapes = _as_list(slot.out_shapes(inputs))
        outs = [torch.empty(s, dtype=dtype, device=device) for s in out_shapes]
        try:
            prepared = None
            if slot.invoke_prepare is not None:
                if prepare is None:
                    raise RuntimeError(
                        f"slot {slot.name!r} is a (prepare, forward) slot but no 'prepare' callable was provided"
                    )
                prepared = slot.invoke_prepare(prepare, inputs)  # runs the miner's weight-prep
            slot.invoke_entry(entry, inputs, outs, prepared)
        except Exception as exc:  # noqa: BLE001 - report kernel failure as a fail
            results.append(
                ShapeResult(shape=shape, dtype=_name(dtype), passed=False,
                            max_abs_err=float("inf"), max_rel_err=float("inf"), pass_ratio=0.0,
                            detail=f"kernel raised: {type(exc).__name__}: {exc}")
            )
            continue

        passed = True
        max_abs = 0.0
        max_rel = 0.0
        min_score_seen = 1.0
        metric = "ratio"
        details: list[str] = []
        for j, (o, e) in enumerate(zip(outs, expected)):
            p, ma, mr, score, detail, metric = _compare(o, e, atol=tol.atol, rtol=tol.rtol, correctness=slot.correctness)
            passed = passed and p
            max_abs = max(max_abs, ma)
            max_rel = max(max_rel, mr)
            min_score_seen = min(min_score_seen, score)
            if detail:
                details.append(f"out[{j}]: {detail}" if len(outs) > 1 else detail)
        results.append(
            ShapeResult(shape=shape, dtype=_name(dtype), passed=passed,
                        max_abs_err=max_abs, max_rel_err=max_rel, pass_ratio=min_score_seen,
                        detail="; ".join(details), metric=metric)
        )

    return VerifyResult(
        slot=slot.name,
        dtype=_name(dtype),
        passed=all(r.passed for r in results) and len(results) > 0,
        shape_results=results,
    )


def _name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def format_verify(result: VerifyResult) -> str:
    lines = [f"[{'PASS' if result.passed else 'FAIL'}] {result.slot} dtype={result.dtype}"]
    for r in result.shape_results:
        status = "ok " if r.passed else "FAIL"
        if r.metric == "cosine":
            score = f" cos={r.pass_ratio:.5f}"
        else:
            score = "" if r.pass_ratio >= 1.0 else f" ratio={r.pass_ratio:.4f}"
        lines.append(
            f"  {status} shape={r.shape} max_abs={r.max_abs_err:.3e} max_rel={r.max_rel_err:.3e}{score}"
            + (f"  {r.detail}" if r.detail else "")
        )
    return "\n".join(lines)
