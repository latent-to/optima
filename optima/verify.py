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
pipeline still runs the end-to-end gate. To stop a kernel from special-casing the fixed
verification inputs, the input VALUES vary with ``seed`` and, when ``jitter_seed`` is
set (the CLI path does this per run), the COUNT dimensions (num_tokens / batch / ctx)
are perturbed too — so a kernel can't hard-code the exact verify shapes. Feature dims
(hidden / head_dim) are left intact since kernels legitimately specialize on them; the
end-to-end gate on fresh prompts is the backstop against shape-branching there.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import torch

from optima.capabilities import (
    CONTEXT_FIELDS,
    CallDescriptor,
    msa_prefill_call_descriptor,
)
from optima.registry import Eligibility
from optima.slots import SlotSpec
from optima.tensor_spec import (
    allocate_output_spec,
    tensor_storage_binding,
    validate_output_allocation,
    validate_tensor_binding,
)


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
    # Number of successful CUDA-graph replays checked against the trusted
    # reference for this shape.  Zero means this was an eager-only verification
    # (including every CPU run), not that graph correctness was established.
    graph_replays: int = 0
    # False means the validator proved this catalog shape lies outside the
    # variant's declared domain and did not invoke miner code.
    applicable: bool = True


@dataclass
class VerifyResult:
    slot: str
    dtype: str
    passed: bool
    shape_results: list[ShapeResult]
    # ``passed`` remains the ordinary numerical verdict so CPU verification keeps
    # its historical meaning.  A crown/qualification path for a graph-safe bundle
    # must additionally require ``graph_verified`` whenever ``graph_required`` is
    # true; this prevents a CPU-only run from masquerading as graph proof.
    graph_required: bool = False
    graph_verified: bool = False
    # Zero denotes an unfiltered legacy run. A filtered variant must exercise at
    # least one validator-controlled shape; arena-grade coverage policy is later.
    coverage_required: int = 0
    # True means this variant targets a different invariant arena context
    # (dtype/architecture/phase/layout/TP/etc.), not that it matched this arena
    # but escaped the validator's shape probes. Bundle verification may report
    # such a row N/A when another sibling variant is applicable.
    context_inapplicable: bool = False
    # False means the declared finite domain exceeded the validator's bounded
    # semantic-probe budget. Partial enumeration is never a passing verdict.
    domain_coverage_complete: bool = True
    domain_coverage_detail: str = ""

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.shape_results if r.applicable and not r.passed)

    @property
    def num_applicable(self) -> int:
        return sum(1 for r in self.shape_results if r.applicable)

    @property
    def num_not_applicable(self) -> int:
        return len(self.shape_results) - self.num_applicable

    @property
    def coverage_sufficient(self) -> bool:
        return self.coverage_required == 0 or self.num_applicable >= self.coverage_required

    @property
    def fully_verified(self) -> bool:
        """Whether every gate requested by this verification run was proven."""
        return self.passed and (not self.graph_required or self.graph_verified)


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
    if correctness.mode == "topk_overlap":
        # Selection metric: only WHICH top-k are picked matters (not the score values), so
        # masked-out positions may legitimately be -inf. NaN/+inf are never legitimate
        # block scores, however: rejecting them is also load-bearing for graph replay,
        # whose NaN poison must prove that every output cell was rewritten.
        if torch.isnan(a).any() or torch.isposinf(a).any():
            return (
                False,
                float("inf"),
                float("inf"),
                0.0,
                "actual has NaN or +inf block scores",
                "overlap",
            )
        k = correctness.top_k
        ta = a.topk(k, dim=-1).indices
        te = e.topk(k, dim=-1).indices
        overlap = (ta.unsqueeze(-1) == te.unsqueeze(-2)).any(dim=-1).float().mean(dim=-1)
        score = float(overlap.mean())
        passed = score >= correctness.min_overlap
        detail = "" if passed else f"topk_overlap {score:.4f} < min_overlap {correctness.min_overlap}"
        return passed, 0.0, 0.0, score, detail, "overlap"
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


@dataclass
class _OutputCheck:
    passed: bool
    max_abs: float
    max_rel: float
    min_score: float
    detail: str
    metric: str


def _compare_outputs(outs: list[torch.Tensor], expected: list[torch.Tensor], *, tol,
                     correctness) -> _OutputCheck:
    """Compare every declared output and retain the worst result.

    Kept separate from ``verify_entry`` because CUDA-graph replay must apply the
    exact same comparator as eager verification on every replay.  A different or
    weaker graph comparator would recreate the very eager-vs-captured gap this gate
    is intended to close.
    """
    if len(outs) != len(expected):
        return _OutputCheck(
            False, float("inf"), float("inf"), 0.0,
            f"output count mismatch {len(outs)} vs {len(expected)}", "ratio",
        )

    passed = True
    max_abs = 0.0
    max_rel = 0.0
    min_score_seen = 1.0
    metric = "ratio"
    details: list[str] = []
    for j, (out, reference) in enumerate(zip(outs, expected)):
        p, ma, mr, score, detail, metric = _compare(
            out, reference, atol=tol.atol, rtol=tol.rtol, correctness=correctness
        )
        passed = passed and p
        max_abs = max(max_abs, ma)
        max_rel = max(max_rel, mr)
        min_score_seen = min(min_score_seen, score)
        if detail:
            details.append(f"out[{j}]: {detail}" if len(outs) > 1 else detail)
    return _OutputCheck(
        passed, max_abs, max_rel, min_score_seen, "; ".join(details), metric
    )


class _GraphBackend(Protocol):
    """Small adapter so graph orchestration can be unit-tested without a GPU."""

    def warmup(self, fn: Callable[[], None]) -> None: ...

    def capture(self, fn: Callable[[], None]): ...

    def replay(self, graph) -> None: ...

    def synchronize(self) -> None: ...


class _CudaGraphBackend:
    """Real PyTorch CUDA-graph capture backend.

    Warmup happens on a side stream, as required for graph-safe lazy/JIT kernels,
    before genuine ``torch.cuda.CUDAGraph`` capture.  Candidate Python runs during
    capture but not replay, which is load-bearing: a branch on
    ``torch.cuda.is_current_stream_capturing()`` is frozen into the graph and its
    captured behavior is what the replay comparisons grade.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            # torch.cuda.graph's implicit stream is class-global and therefore
            # can be stranded on the first GPU used in this process. Pin an
            # explicit stream to the candidate output device instead.
            self.capture_stream = torch.cuda.Stream(device=self.device)

    def warmup(self, fn: Callable[[], None]) -> None:
        with torch.cuda.device(self.device):
            current = torch.cuda.current_stream(self.device)
            warmup_stream = torch.cuda.Stream(device=self.device)
            warmup_stream.wait_stream(current)
            with torch.cuda.stream(warmup_stream):
                fn()
            current.wait_stream(warmup_stream)
            torch.cuda.synchronize(self.device)

    def capture(self, fn: Callable[[], None]):
        with torch.cuda.device(self.device):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=self.capture_stream):
                fn()
            return graph

    def replay(self, graph) -> None:
        with torch.cuda.device(self.device):
            graph.replay()

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device)


_DEFAULT_GRAPH_REPLAYS = 3


def _poison_outputs(outs: list[torch.Tensor], replay: int) -> None:
    """Overwrite outputs before each replay so a partial/no-op graph cannot pass.

    The poison is intentionally changed per replay for integral outputs.  Floating
    outputs use NaN, which every comparator rejects unless the replay overwrites the
    relevant cells, including score sheets graded by ``topk_overlap``.
    """
    with torch.no_grad():
        for out in outs:
            if out.dtype.is_floating_point or out.dtype.is_complex:
                out.fill_(float("nan"))
            elif out.dtype == torch.bool:
                out.fill_(bool(replay % 2))
            else:
                info = torch.iinfo(out.dtype)
                out.fill_(info.max - (replay % 17))


@dataclass
class _GraphCheck:
    check: _OutputCheck
    replays: int


@dataclass(frozen=True)
class _GraphReplayCase:
    """One fresh logical request copied into already-captured tensor addresses."""

    inputs: dict
    expected: list[torch.Tensor]


def _clone_tensor_inputs(inputs: dict) -> dict:
    """Snapshot built-in slot inputs without retaining candidate-visible storage.

    Slot generators currently return disjoint tensors plus immutable Python scalars.
    This helper intentionally does not claim to preserve arbitrary container aliasing;
    a future typed-input ABI must make that policy explicit before accepting it.
    """

    return {
        name: value.detach().clone() if torch.is_tensor(value) else value
        for name, value in inputs.items()
    }


def _input_bindings(inputs: dict) -> dict:
    """Retain exact pre-candidate tensor/storage identities by input name."""

    return {
        name: tensor_storage_binding(value)
        for name, value in inputs.items()
        if torch.is_tensor(value)
    }


def _input_mutation_detail(actual: dict, trusted: dict, bindings: dict) -> str:
    """Return the first tensor input changed by candidate code, else ``""``."""

    for name, expected in trusted.items():
        if not torch.is_tensor(expected):
            continue
        value = actual.get(name)
        if not torch.is_tensor(value):
            return f"input {name!r} ceased to be a tensor"
        binding = bindings.get(name)
        if binding is None:
            return f"input {name!r} has no validator-owned binding"
        try:
            validate_tensor_binding(value, binding, name=f"input {name!r}")
        except ValueError as exc:
            return str(exc)
        if (value.shape != expected.shape or value.dtype != expected.dtype
                or value.device != expected.device):
            return f"input {name!r} metadata changed"
        if not torch.equal(value, expected):
            return f"input {name!r} was mutated"
    return ""


def _restore_tensor_inputs(actual: dict, trusted: dict, bindings: dict) -> None:
    """Copy trusted values into the original tensor objects captured by CUDA graphs."""

    with torch.no_grad():
        for name, expected in trusted.items():
            if not torch.is_tensor(expected):
                continue
            value = actual.get(name)
            if not torch.is_tensor(value):
                raise RuntimeError(f"input {name!r} ceased to be a tensor")
            binding = bindings.get(name)
            if binding is None:
                raise RuntimeError(f"input {name!r} has no validator-owned binding")
            validate_tensor_binding(value, binding, name=f"input {name!r}")
            if (value.shape != expected.shape or value.dtype != expected.dtype
                    or value.device != expected.device):
                raise RuntimeError(f"input {name!r} metadata changed")
            value.copy_(expected)


def _graph_case_inputs(slot: SlotSpec, trusted: dict, generated: dict) -> dict:
    """Merge fresh request tensors with the trusted capture-static input state."""

    names = slot.graph_dynamic_inputs
    if not names:
        raise RuntimeError(
            f"slot {slot.name!r} declares no graph-dynamic tensor inputs"
        )
    if len(set(names)) != len(names):
        raise RuntimeError(f"slot {slot.name!r} repeats a graph-dynamic input name")
    logical = dict(trusted)
    for name in names:
        base = trusted.get(name)
        fresh = generated.get(name)
        if not torch.is_tensor(base) or not torch.is_tensor(fresh):
            raise RuntimeError(
                f"slot {slot.name!r} graph-dynamic input {name!r} is not a tensor"
            )
        if (fresh.shape != base.shape or fresh.dtype != base.dtype
                or fresh.device != base.device):
            raise RuntimeError(
                f"slot {slot.name!r} graph-dynamic input {name!r} changed metadata"
            )
        if torch.equal(fresh, base):
            raise RuntimeError(
                f"slot {slot.name!r} generator did not vary graph-dynamic input {name!r}"
            )
        logical[name] = fresh.detach().clone()
    return logical


def _verify_graph_replays(
    slot: SlotSpec,
    entry: Callable[..., None],
    inputs: dict,
    output_contract,
    allocation,
    prepared,
    trusted_inputs: dict,
    input_bindings: dict,
    replay_cases: list[_GraphReplayCase],
    *,
    tol,
    replay_count: int = _DEFAULT_GRAPH_REPLAYS,
    backend: Optional[_GraphBackend] = None,
    fallback_dtype: torch.dtype,
    fallback_device: str | torch.device,
) -> _GraphCheck:
    """Capture the candidate once and grade multiple genuine graph replays.

    ``backend`` is injectable solely to exercise the orchestration with CPU tensors
    in unit tests.  Production callers omit it and therefore always use
    ``torch.cuda.CUDAGraph``.
    """
    if replay_count < 2:
        raise ValueError("CUDA graph verification requires at least two replays")
    if len(replay_cases) != replay_count:
        raise ValueError("one fresh trusted case is required per graph replay")
    outs = allocation.outputs
    graph_backend = backend or _CudaGraphBackend(outs[0].device)

    def invoke() -> None:
        slot.invoke_entry(entry, inputs, outs, prepared)

    def validate_outputs() -> None:
        validate_output_allocation(
            output_contract,
            allocation,
            fallback_dtype=fallback_dtype,
            fallback_device=fallback_device,
            inputs=(value for value in inputs.values() if torch.is_tensor(value)),
        )

    try:
        _restore_tensor_inputs(inputs, trusted_inputs, input_bindings)
        graph_backend.warmup(invoke)
        graph_backend.synchronize()
        validate_outputs()
        mutation = _input_mutation_detail(inputs, trusted_inputs, input_bindings)
        if mutation:
            raise RuntimeError(mutation)
    except Exception as exc:  # noqa: BLE001 - candidate warmup failure is a verdict
        return _GraphCheck(
            _OutputCheck(False, float("inf"), float("inf"), 0.0,
                         f"cuda graph warmup raised: {type(exc).__name__}: {exc}", "ratio"),
            0,
        )
    try:
        _restore_tensor_inputs(inputs, trusted_inputs, input_bindings)
        graph = graph_backend.capture(invoke)
        graph_backend.synchronize()
        validate_outputs()
        mutation = _input_mutation_detail(inputs, trusted_inputs, input_bindings)
        if mutation:
            raise RuntimeError(mutation)
    except Exception as exc:  # noqa: BLE001 - a graph_safe claim must actually capture
        return _GraphCheck(
            _OutputCheck(False, float("inf"), float("inf"), 0.0,
                         f"cuda graph capture raised: {type(exc).__name__}: {exc}", "ratio"),
            0,
        )

    max_abs = 0.0
    max_rel = 0.0
    min_score = 1.0
    metric = "ratio"
    completed = 0
    for replay, case in enumerate(replay_cases):
        try:
            _restore_tensor_inputs(inputs, case.inputs, input_bindings)
            _poison_outputs(outs, replay)
            # Be explicit about the poison-before-replay happens-before edge.  This
            # is a correctness gate, not a benchmark; the synchronization is desired.
            graph_backend.synchronize()
            graph_backend.replay(graph)
            graph_backend.synchronize()
            validate_outputs()
            mutation = _input_mutation_detail(inputs, case.inputs, input_bindings)
            if mutation:
                raise RuntimeError(mutation)
        except Exception as exc:  # noqa: BLE001 - replay failure is a failed claim
            return _GraphCheck(
                _OutputCheck(
                    False, float("inf"), float("inf"), 0.0,
                    f"cuda graph replay[{replay}] raised: {type(exc).__name__}: {exc}",
                    "ratio",
                ),
                completed,
            )

        completed = replay + 1
        current = _compare_outputs(
            outs, case.expected, tol=tol, correctness=slot.correctness
        )
        max_abs = max(max_abs, current.max_abs)
        max_rel = max(max_rel, current.max_rel)
        min_score = min(min_score, current.min_score)
        metric = current.metric
        if not current.passed:
            detail = current.detail or "output mismatch"
            return _GraphCheck(
                _OutputCheck(False, max_abs, max_rel, min_score,
                             f"cuda graph replay[{replay}]: {detail}", metric),
                completed,
            )

    return _GraphCheck(
        _OutputCheck(True, max_abs, max_rel, min_score, "", metric), completed
    )


# Count-like shape keys safe to jitter (varying these doesn't break a kernel that
# legitimately specializes on the feature dims like hidden / head_dim / inter).
_JITTER_KEYS = ("num_tokens", "batch", "ctx", "q_len", "prefix_blocks")


def _jitter_shapes(shapes: list[dict], seed: int) -> list[dict]:
    """Perturb the count dimensions of each shape deterministically from ``seed`` so the
    verify shapes vary per run — a kernel can't hard-code the exact verification token
    counts. Feature dims are untouched; counts stay >= 1."""
    import random

    rng = random.Random(seed)
    out: list[dict] = []
    for sh in shapes:
        s = dict(sh)
        for k in _JITTER_KEYS:
            if s.get("causal_probe") is True and k == "q_len":
                # This semantic probe assigns one orthogonal feature dimension
                # per query row. Its catalog q_len is part of the adversary, not
                # a free count dimension; varying values and prefix length remain.
                continue
            if k in s and isinstance(s[k], int):
                # The attention generators reserve row 0 as a full-length semantic
                # probe. Keeping batch >= 2 leaves another request whose seq_lens can
                # vary across graph replays instead of producing vacuous fixed input.
                floor = 2 if k == "batch" else 1
                s[k] = max(
                    floor,
                    s[k] + rng.randint(-1, 3) + (s[k] // 3) * rng.randint(0, 1),
                )
        out.append(s)
    return out


_MSA_PROBE_MAX_HEAD_DIM = 512
_MSA_PROBE_MAX_BLOCK_SIZE = 4096
_MSA_PROBE_MAX_Q_LEN = 1024
_MSA_PROBE_MAX_KV_LEN = 32768
_MSA_PROBE_MAX_MATMUL_WORK = 300_000_000
_MSA_PROBE_MAX_TOTAL_WORK = 600_000_000
_MSA_MAX_CANDIDATE_COMBINATIONS = 64
_MSA_MAX_SYNTHESIZED_SHAPES = 32
_MSA_PREFILL_SHAPE_FIELDS = frozenset(
    {"q_len", "prefix_blocks", "head_dim", "block_size"}
)
_MSA_PREFILL_INPUT_FIELDS = frozenset(
    {"q", "index_k", "prefix_len", "scale", "block_size"}
)
_MSA_SYNTHESIZED_CAPABILITY_FIELDS = frozenset(
    {"head_dim", "last_dim", "block_size", "q_len", "num_tokens", "kv_len"}
)


def _has_msa_prefill_probe_schema(
    slot: SlotSpec, eligibility: Eligibility, catalog_shapes: list[dict]
) -> bool:
    """Recognize the semantic probe schema without consulting slot identity."""

    constrained = eligibility.capabilities.constrained_fields
    has_legacy_shape_bound = any(
        value is not None
        for value in (
            eligibility.max_last_dim,
            eligibility.min_num_tokens,
            eligibility.max_num_tokens,
        )
    )
    return (
        slot.correctness.mode == "topk_overlap"
        and (
            bool(constrained & _MSA_SYNTHESIZED_CAPABILITY_FIELDS)
            or has_legacy_shape_bound
        )
        and bool(catalog_shapes)
        and all(_MSA_PREFILL_SHAPE_FIELDS <= set(shape) for shape in catalog_shapes)
    )


def _has_msa_prefill_call_contract(slot: SlotSpec, inputs: dict) -> bool:
    """Recognize the canonical score-sheet call from validator-owned values."""

    return (
        slot.correctness.mode == "topk_overlap"
        and _MSA_PREFILL_INPUT_FIELDS <= set(inputs)
        and torch.is_tensor(inputs["q"])
        and inputs["q"].dim() == 2
        and torch.is_tensor(inputs["index_k"])
        and inputs["index_k"].dim() == 2
    )


def _msa_shape_descriptor(
    slot: SlotSpec,
    shape: dict,
    *,
    dtype: torch.dtype,
    architecture: Optional[str],
    tp_size: Optional[int],
    world_size: Optional[int],
) -> CallDescriptor | None:
    """Describe an MSA probe without allocating its potentially large tensors."""

    q_len = int(shape["q_len"])
    head_dim = int(shape["head_dim"])
    block_size = int(shape["block_size"])
    if min(q_len, head_dim, block_size) < 1:
        return None
    prefix_len = shape.get("prefix_len_override")
    if prefix_len is None:
        prefix_blocks = max(int(shape.get("prefix_blocks", 12)), 12)
        prefix_len = prefix_blocks * block_size + 39
    prefix_len = int(prefix_len)
    kv_len = prefix_len + q_len
    if prefix_len < 0 or kv_len < q_len:
        return None
    return msa_prefill_call_descriptor(
        dtype=_name(dtype),
        architecture=architecture,
        head_dim=head_dim,
        block_size=block_size,
        q_len=q_len,
        kv_len=kv_len,
        top_k=int(slot.correctness.top_k),
        num_kv_heads=1,
        tp_size=tp_size,
        world_size=world_size,
    )


def _msa_numeric_domain_values(
    eligibility: Eligibility,
    fields: set[str],
    defaults: set[int],
    *,
    limit: int,
    legacy_minimum: int | None = None,
    legacy_maximum: int | None = None,
) -> tuple[list[int], list[int], bool, bool]:
    """Return safe validator candidates, with domain-derived values first."""

    derived: set[int] = set()
    constrained = False
    outside_limit = False
    for predicate in eligibility.capabilities.predicates:
        if predicate.field not in fields:
            continue
        constrained = True
        if predicate.allowed:
            values = {
                int(value)
                for value in predicate.allowed
                if isinstance(value, int) and not isinstance(value, bool)
            }
            outside_limit = outside_limit or any(
                value < 1 or value > limit for value in values
            )
            derived.update(values)
            continue
        lo = 1 if predicate.minimum is None else predicate.minimum
        declared_hi = predicate.maximum
        outside_limit = outside_limit or lo < 1 or lo > limit
        outside_limit = outside_limit or declared_hi is None or declared_hi > limit
        hi = limit if declared_hi is None else min(declared_hi, limit)
        if lo <= hi:
            derived.update((lo, hi, (lo + hi) // 2))
    if legacy_minimum is not None:
        constrained = True
        outside_limit = outside_limit or not 1 <= legacy_minimum <= limit
        derived.add(legacy_minimum)
    if legacy_maximum is not None:
        constrained = True
        outside_limit = outside_limit or not 1 <= legacy_maximum <= limit
        derived.add(legacy_maximum)
    elif legacy_minimum is not None:
        # A legacy lower bound with no upper bound denotes an unbounded live
        # domain, just like a named {min: ...} predicate. The bounded verifier
        # may sample it, but may not call that complete qualification evidence.
        outside_limit = True
    derived = {value for value in derived if 1 <= value <= limit}
    ordinary = {value for value in defaults if 1 <= value <= limit} - derived
    return sorted(derived), sorted(ordinary), constrained, outside_limit


def _synthesize_msa_capability_shapes(
    slot: SlotSpec,
    eligibility: Eligibility,
    catalog_shapes: list[dict],
    *,
    dtype: torch.dtype,
    architecture: Optional[str],
    tp_size: Optional[int],
    world_size: Optional[int],
) -> tuple[list[dict], bool, str]:
    """Create bounded semantic probes when a declared MSA domain misses the catalog.

    This is not arena-grade range coverage. It makes new exact/ranged shape domains
    ingestible without an Optima edit while retaining a strict allocation/work ceiling;
    the later ArenaProfile layer owns workload distributions and larger probes.
    """

    catalog_matches = [
        shape
        for shape in catalog_shapes
        if (descriptor := _msa_shape_descriptor(
            slot,
            shape,
            dtype=dtype,
            architecture=architecture,
            tp_size=tp_size,
            world_size=world_size,
        )) is not None and eligibility.match(descriptor).accepted
    ]

    head_defaults = {int(shape["head_dim"]) for shape in catalog_shapes}
    block_defaults = {int(shape["block_size"]) for shape in catalog_shapes}
    q_defaults = {int(shape["q_len"]) for shape in catalog_shapes}
    head_derived, head_ordinary, head_constrained, head_outside = (
        _msa_numeric_domain_values(
        eligibility,
        {"head_dim", "last_dim"},
        head_defaults,
        limit=_MSA_PROBE_MAX_HEAD_DIM,
        legacy_maximum=eligibility.max_last_dim,
        )
    )
    block_derived, block_ordinary, block_constrained, block_outside = (
        _msa_numeric_domain_values(
        eligibility,
        {"block_size"},
        block_defaults,
        limit=_MSA_PROBE_MAX_BLOCK_SIZE,
        )
    )
    q_derived, q_ordinary, q_constrained, q_outside = _msa_numeric_domain_values(
        eligibility,
        {"q_len", "num_tokens"},
        q_defaults,
        limit=_MSA_PROBE_MAX_Q_LEN,
        legacy_minimum=eligibility.min_num_tokens,
        legacy_maximum=eligibility.max_num_tokens,
    )
    kv_derived, _kv_ordinary, kv_constrained, kv_outside = (
        _msa_numeric_domain_values(
        eligibility,
        {"kv_len"},
        set(),
        limit=_MSA_PROBE_MAX_KV_LEN,
        )
    )
    if not (head_constrained or block_constrained or q_constrained or kv_constrained):
        return [], True, ""
    if head_outside or block_outside or q_outside or kv_outside:
        return (
            [],
            False,
            "declared MSA capability domain exceeds the bounded probe limits",
        )

    def _dimension_values(
        derived: list[int],
        ordinary: list[int],
        constrained: bool,
        field: str,
    ) -> list[int]:
        if constrained and derived:
            return derived
        if catalog_matches:
            values = sorted({int(shape[field]) for shape in catalog_matches})
            return values[:1]
        return ordinary[:1]

    heads = _dimension_values(
        head_derived, head_ordinary, head_constrained, "head_dim"
    )
    blocks = _dimension_values(
        block_derived, block_ordinary, block_constrained, "block_size"
    )
    q_lengths = _dimension_values(
        q_derived, q_ordinary, q_constrained, "q_len"
    )
    kv_lengths = kv_derived
    if not heads or not blocks or not q_lengths or (kv_constrained and not kv_lengths):
        return [], True, ""

    kv_factor = len(kv_lengths) if kv_constrained else 1
    candidate_combinations = len(heads) * len(blocks) * len(q_lengths) * kv_factor
    if candidate_combinations > _MSA_MAX_CANDIDATE_COMBINATIONS:
        return (
            [],
            False,
            f"declared MSA capability cross-product has {candidate_combinations} "
            f"probe combinations; limit is {_MSA_MAX_CANDIDATE_COMBINATIONS}",
        )

    synthesized: list[dict] = []
    seen: set[tuple[int, int, int, int]] = set()
    catalog_prefixes: dict[tuple[int, int, int], int] = {}
    for shape in catalog_shapes:
        q_len = int(shape["q_len"])
        head_dim = int(shape["head_dim"])
        block_size = int(shape["block_size"])
        prefix_len = shape.get("prefix_len_override")
        if prefix_len is None:
            prefix_len = max(int(shape.get("prefix_blocks", 12)), 12) * block_size + 39
        prefix_len = int(prefix_len)
        seen.add((q_len, head_dim, block_size, prefix_len))
        catalog_prefixes.setdefault((q_len, head_dim, block_size), prefix_len)
    total_work = 0
    for head_dim in heads:
        for block_size in blocks:
            for q_len in q_lengths:
                # Prefix length is a semantic property of each block-size
                # combination. Reusing the byte/token prefix from a catalog case
                # with a smaller block can leave <= top_k visible blocks and make
                # the selection grade vacuous. Scale it independently here; an
                # identical catalog case is already removed by ``seen`` below.
                default_prefix = catalog_prefixes.get(
                    (q_len, head_dim, block_size),
                    max(12, int(slot.correctness.top_k) + 4) * block_size + 39,
                )
                candidate_kv = kv_lengths if kv_constrained else [default_prefix + q_len]
                for kv_len in candidate_kv:
                    prefix_len = kv_len - q_len
                    visible_blocks = (prefix_len + block_size) // block_size
                    if prefix_len < 0 or visible_blocks <= int(slot.correctness.top_k):
                        return (
                            [],
                            False,
                            "an MSA capability combination cannot form a "
                            "non-vacuous top-k probe",
                        )
                    key = (q_len, head_dim, block_size, prefix_len)
                    if key in seen:
                        continue
                    shape = {
                        "q_len": q_len,
                        "prefix_blocks": max(12, prefix_len // block_size),
                        "prefix_len_override": prefix_len,
                        "head_dim": head_dim,
                        "block_size": block_size,
                    }
                    descriptor = _msa_shape_descriptor(
                        slot,
                        shape,
                        dtype=dtype,
                        architecture=architecture,
                        tp_size=tp_size,
                        world_size=world_size,
                    )
                    if descriptor is None or not eligibility.match(descriptor).accepted:
                        return (
                            [],
                            False,
                            "an MSA capability combination could not be represented "
                            "by the canonical call descriptor",
                        )
                    work = q_len * kv_len * head_dim
                    probe_count = 2 if q_len > 1 else 1
                    if work > _MSA_PROBE_MAX_MATMUL_WORK:
                        return (
                            [],
                            False,
                            "an MSA capability probe exceeds the per-shape work limit",
                        )
                    if (
                        len(synthesized) + probe_count
                        > _MSA_MAX_SYNTHESIZED_SHAPES
                        or total_work + work * probe_count > _MSA_PROBE_MAX_TOTAL_WORK
                    ):
                        return (
                            [],
                            False,
                            "declared MSA capability domain exceeds the total probe budget",
                        )
                    seen.add(key)
                    synthesized.append(shape)
                    total_work += work
                    if q_len > 1:
                        synthesized.append({**shape, "causal_probe": True})
                        total_work += work
    return synthesized, True, ""


def verify_entry(
    slot: SlotSpec,
    entry: Callable[..., None],
    *,
    prepare: Optional[Callable] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[str] = None,
    seed: int = 0,
    shapes: Optional[list[dict]] = None,
    jitter_seed: Optional[int] = None,
    graph_safe: Optional[bool] = None,
    graph_replays: int = _DEFAULT_GRAPH_REPLAYS,
    eligibility: Optional[Eligibility] = None,
    architecture: Optional[str] = None,
    tp_size: Optional[int] = None,
    world_size: Optional[int] = None,
    _graph_backend: Optional[_GraphBackend] = None,
) -> VerifyResult:
    """Verify a miner ``entry`` against the slot's reference.

    ``entry`` is called via ``slot.invoke_entry(entry, inputs, outs, prepared)`` and
    must write its result into the validator-allocated tensors in ``outs``. For a
    *(prepare, forward)* slot (``slot.prepare`` set, e.g. ``moe.fused_experts``) pass
    the miner's ``prepare`` callable too — it runs once on the raw weights and its
    result is handed to ``entry`` as ``prepared`` (otherwise ``prepared`` is None).

    On CUDA, op slots are graph-verified by default because their serving seam is
    always captured.  Block slots are graph-verified when the caller passes their
    declared ``graph_safe=True`` metadata.  CPU runs retain the eager numerical gate
    but return ``graph_required=True, graph_verified=False`` when graph proof was
    requested.  With ``eligibility``, validator code describes every generated
    call before invocation: off-domain shapes are reported N/A without entering
    miner code, and a domain matching zero catalog shapes fails verification.
    ``_graph_backend`` is a private CPU-test hook; production must omit it.
    """
    if getattr(slot, "kind", None) == "collective":
        raise ValueError(
            f"slot {slot.name!r} is a collective slot — verify it distributed with "
            "optima.verify_collective.verify_collective, not the single-process verify_entry"
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    graph_required = slot.kind == "op" if graph_safe is None else bool(graph_safe)
    graph_capable_run = str(device).startswith("cuda") or _graph_backend is not None
    if graph_required and graph_replays < 2:
        raise ValueError("CUDA graph verification requires at least two replays")
    tol = slot.tolerance_for(dtype)
    catalog_shapes = list(shapes) if shapes is not None else list(slot.shapes)
    domain_coverage_complete = True
    domain_coverage_detail = ""
    if eligibility is not None and _has_msa_prefill_probe_schema(
        slot, eligibility, catalog_shapes
    ):
        resolved_arch = architecture or _device_architecture(device)
        synthesized, domain_coverage_complete, domain_coverage_detail = (
            _synthesize_msa_capability_shapes(
                slot, eligibility, catalog_shapes, dtype=dtype,
                architecture=resolved_arch, tp_size=tp_size, world_size=world_size,
            )
        )
        catalog_shapes.extend(synthesized)
    test_shapes = catalog_shapes
    if jitter_seed is not None:
        test_shapes = _jitter_shapes(catalog_shapes, jitter_seed)

    results: list[ShapeResult] = []
    context_blocked: list[bool] = []
    for i, (catalog_shape, jittered_shape) in enumerate(
        zip(catalog_shapes, test_shapes)
    ):
        shape = jittered_shape
        inputs = slot.make_inputs(dtype=dtype, device=device, seed=seed + i, **shape)
        if eligibility is not None:
            descriptor = _verification_call_descriptor(
                slot,
                inputs,
                dtype=dtype,
                device=device,
                architecture=architecture,
                tp_size=tp_size,
                world_size=world_size,
            )
            match = eligibility.match(descriptor)
            if not match.accepted and shape != catalog_shape:
                # Jitter is an anti-hardcoding challenge, not a license to move a
                # shape-specialized implementation outside its own declared domain.
                # If the perturbed call is out-of-domain but the validator's catalog
                # call is in-domain, qualify the latter. Exact-value variants still
                # receive fresh input values/seeds; ranges retain jitter whenever the
                # perturbation remains inside the range.
                catalog_inputs = slot.make_inputs(
                    dtype=dtype,
                    device=device,
                    seed=seed + i,
                    **catalog_shape,
                )
                catalog_descriptor = _verification_call_descriptor(
                    slot,
                    catalog_inputs,
                    dtype=dtype,
                    device=device,
                    architecture=architecture,
                    tp_size=tp_size,
                    world_size=world_size,
                )
                catalog_match = eligibility.match(catalog_descriptor)
                if catalog_match.accepted:
                    shape = catalog_shape
                    inputs = catalog_inputs
                    match = catalog_match
            if not match.accepted:
                static_context_fields = CONTEXT_FIELDS | {
                    "ep_size",
                    "tp_size",
                    "world_size",
                }
                context_blocked.append(
                    any(
                        mismatch.field in static_context_fields
                        for mismatch in match.mismatches
                    )
                )
                reasons = "; ".join(
                    f"{m.field} {m.reason}: expected {m.expected}"
                    + ("" if m.actual is None else f", got {m.actual!r}")
                    for m in match.mismatches
                )
                results.append(ShapeResult(
                    shape=shape,
                    dtype=_name(dtype),
                    passed=True,
                    max_abs_err=0.0,
                    max_rel_err=0.0,
                    detail=f"validator N/A (outside declared capability domain): {reasons}",
                    metric="n/a",
                    applicable=False,
                ))
                continue
        context_blocked.append(False)
        # The trusted reference is derived from storage the candidate never sees,
        # before either prepare or entry can mutate live inputs.  These receipts are
        # still diagnostic (the candidate shares this worker process); pristine-T in
        # qualification is the eventual external authority.
        trusted_inputs = _clone_tensor_inputs(inputs)
        input_bindings = _input_bindings(inputs)
        expected = [
            output.detach().clone()
            for output in _as_list(slot.invoke_reference(trusted_inputs))
        ]
        replay_cases: list[_GraphReplayCase] = []
        graph_case_error = ""
        if graph_required and graph_capable_run:
            for replay in range(graph_replays):
                last_error = ""
                for attempt in range(8):
                    fresh = slot.make_inputs(
                        dtype=dtype,
                        device=device,
                        seed=(
                            seed + i + 104_729 * (replay + 1)
                            + 1_000_003 * attempt
                        ),
                        **shape,
                    )
                    try:
                        logical = _graph_case_inputs(slot, trusted_inputs, fresh)
                    except RuntimeError as exc:
                        last_error = str(exc)
                        continue
                    replay_expected = [
                        output.detach().clone()
                        for output in _as_list(slot.invoke_reference(logical))
                    ]
                    replay_cases.append(_GraphReplayCase(logical, replay_expected))
                    break
                else:
                    graph_case_error = last_error or "could not generate fresh graph inputs"
                    break
        # Allocate from the same typed contract used by the live arena binding.
        # Legacy slots resolve ``out_shapes`` to inherited-dtype contiguous tensors,
        # exactly preserving their historical behavior.
        output_contract = slot.output_contract(inputs)
        allocation = allocate_output_spec(
            output_contract,
            fallback_dtype=dtype,
            fallback_device=device,
            inputs=(v for v in inputs.values() if torch.is_tensor(v)),
        )
        outs = allocation.outputs
        if graph_case_error:
            results.append(
                ShapeResult(
                    shape=shape,
                    dtype=_name(dtype),
                    passed=False,
                    max_abs_err=float("inf"),
                    max_rel_err=float("inf"),
                    pass_ratio=0.0,
                    detail=f"graph input refresh unavailable: {graph_case_error}",
                )
            )
            continue
        try:
            prepared = None
            if slot.invoke_prepare is not None:
                if prepare is None:
                    raise RuntimeError(
                        f"slot {slot.name!r} is a (prepare, forward) slot but no 'prepare' callable was provided"
                    )
                prepared = slot.invoke_prepare(prepare, inputs)  # runs the miner's weight-prep
                mutation = _input_mutation_detail(
                    inputs, trusted_inputs, input_bindings
                )
                if mutation:
                    raise RuntimeError(f"prepare {mutation}")
            slot.invoke_entry(entry, inputs, outs, prepared)
            validate_output_allocation(
                output_contract,
                allocation,
                fallback_dtype=dtype,
                fallback_device=device,
                inputs=(value for value in inputs.values() if torch.is_tensor(value)),
            )
            mutation = _input_mutation_detail(inputs, trusted_inputs, input_bindings)
            if mutation:
                raise RuntimeError(mutation)
        except Exception as exc:  # noqa: BLE001 - report kernel failure as a fail
            results.append(
                ShapeResult(shape=shape, dtype=_name(dtype), passed=False,
                            max_abs_err=float("inf"), max_rel_err=float("inf"), pass_ratio=0.0,
                            detail=f"kernel raised: {type(exc).__name__}: {exc}")
            )
            continue

        eager = _compare_outputs(outs, expected, tol=tol, correctness=slot.correctness)
        passed = eager.passed
        max_abs = eager.max_abs
        max_rel = eager.max_rel
        min_score_seen = eager.min_score
        metric = eager.metric
        details = [eager.detail] if eager.detail else []
        checked_replays = 0

        # Do not attempt capture after an eager mismatch: it cannot rescue the
        # candidate, and some broken kernels leave state that only obscures the root
        # error.  Every eager-correct graph-required GPU shape must capture and replay.
        if passed and graph_required and graph_capable_run:
            graph = _verify_graph_replays(
                slot, entry, inputs, output_contract, allocation, prepared,
                trusted_inputs, input_bindings, replay_cases,
                tol=tol,
                replay_count=graph_replays, backend=_graph_backend,
                fallback_dtype=dtype, fallback_device=device,
            )
            checked_replays = graph.replays
            passed = passed and graph.check.passed
            max_abs = max(max_abs, graph.check.max_abs)
            max_rel = max(max_rel, graph.check.max_rel)
            min_score_seen = min(min_score_seen, graph.check.min_score)
            metric = graph.check.metric
            if graph.check.detail:
                details.append(graph.check.detail)
        results.append(
            ShapeResult(shape=shape, dtype=_name(dtype), passed=passed,
                        max_abs_err=max_abs, max_rel_err=max_rel, pass_ratio=min_score_seen,
                        detail="; ".join(details), metric=metric,
                        graph_replays=checked_replays)
        )

    applicable = [result for result in results if result.applicable]
    coverage_required = 1 if eligibility is not None else 0
    coverage_sufficient = (
        len(applicable) >= coverage_required if coverage_required else bool(applicable)
    )
    graph_verified = bool(
        graph_required and graph_capable_run and applicable
        and all(r.passed and r.graph_replays == graph_replays for r in applicable)
    )
    context_inapplicable = bool(
        eligibility is not None
        and not applicable
        and context_blocked
        and all(context_blocked)
    )
    return VerifyResult(
        slot=slot.name,
        dtype=_name(dtype),
        passed=(
            domain_coverage_complete
            and coverage_sufficient
            and all(r.passed for r in applicable)
        ),
        shape_results=results,
        graph_required=graph_required,
        graph_verified=graph_verified,
        coverage_required=coverage_required,
        context_inapplicable=context_inapplicable,
        domain_coverage_complete=domain_coverage_complete,
        domain_coverage_detail=domain_coverage_detail,
    )


def _verification_call_descriptor(
    slot: SlotSpec,
    inputs: dict,
    *,
    dtype: torch.dtype,
    device: str,
    architecture: Optional[str],
    tp_size: Optional[int],
    world_size: Optional[int],
) -> CallDescriptor:
    """Build the same canonical call description as a live arena binding.

    Fields are semantic and therefore never guessed from vaguely similar tensor
    names.  A slot needs an explicit validator mapping before miners can constrain
    its richer dimensions; MSA prefill is the first migrated binding.
    """

    resolved_arch = architecture or _device_architecture(device)
    if not _has_msa_prefill_call_contract(slot, inputs):
        primary = next(
            (
                inputs[name]
                for name in ("x", "q", "input", "input_tensor", "residual", "gemm_out")
                if name in inputs and torch.is_tensor(inputs[name])
            ),
            None,
        )
        fields = {"dtype": _name(dtype), "architecture": resolved_arch}
        if primary is not None and primary.dim() > 0:
            fields["last_dim"] = int(primary.shape[-1])
            if slot.name in {
                "collective.ar_residual_rmsnorm",
                "collective.moe_finalize_ar_rmsnorm",
            }:
                fields["num_tokens"] = int(primary.shape[0])
        return CallDescriptor(fields)
    q = inputs["q"]
    index_k = inputs["index_k"]
    return msa_prefill_call_descriptor(
        dtype=_name(q.dtype),
        architecture=resolved_arch,
        head_dim=int(q.shape[-1]),
        block_size=int(inputs["block_size"]),
        q_len=int(q.shape[0]),
        kv_len=int(index_k.shape[0]),
        top_k=int(slot.correctness.top_k),
        num_kv_heads=1,
        tp_size=tp_size,
        world_size=world_size,
    )


def _device_architecture(device: str) -> Optional[str]:
    resolved = torch.device(device)
    if resolved.type != "cuda" or not torch.cuda.is_available():
        return None
    index = resolved.index
    if index is None:
        index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return f"sm{major}{minor}"


def _direct_aot_prepare_boundary(
    slot: SlotSpec,
    entry: Callable[..., None],
    *,
    prepare_name: Optional[str],
) -> Optional[Callable]:
    """Resolve only the validator-generated lifecycle boundary of a direct entry.

    Direct-AOT rows are forbidden from naming a Python ``prepare`` callable: their
    prepare implementation is assembled from the sealed artifact plan and exposed
    as ``entry.prepare`` by the validator runtime.  Prepare+forward slots must see
    that exact callable or admission fails before any verification launch.
    """

    if prepare_name is not None:
        raise ValueError(
            "direct CuTe AOT rows cannot declare a candidate Python prepare callable"
        )
    prepare = getattr(entry, "prepare", None)
    if prepare is not None and not callable(prepare):
        raise RuntimeError("direct CuTe AOT prepare boundary is not callable")
    if slot.invoke_prepare is not None:
        if prepare is None:
            raise RuntimeError(
                f"slot {slot.name!r} requires a callable validator-generated "
                "direct CuTe AOT prepare boundary"
            )
        return prepare
    return None


def verify_entry_from_source(
    slot_name: str,
    source_path: str,
    entry_name: str,
    *,
    prepare_name: Optional[str] = None,
    dtype_name: str = "bfloat16",
    device: Optional[str] = None,
    seed: int = 0,
    shapes: Optional[list[dict]] = None,
    jitter_seed: Optional[int] = None,
    model_key: Optional[str] = None,
    override_point: Optional[str] = None,
    graph_safe: Optional[bool] = None,
    graph_replays: int = _DEFAULT_GRAPH_REPLAYS,
    eligibility_metadata: Optional[dict] = None,
    manifest_dtypes: tuple[str, ...] = (),
    manifest_architectures: tuple[str, ...] = (),
    tp_size: Optional[int] = None,
    world_size: Optional[int] = None,
    bundle_path: Optional[str] = None,
    variant_name: Optional[str] = None,
) -> VerifyResult:
    """Load the miner module and verify it — module-level + picklable so the CLI can run
    it via ``call_in_subprocess`` in a FRESH process. This keeps the trusted validator/CLI
    process from ever importing miner code (import-time payloads + the kernel run only in the
    throwaway child). It is NOT a security boundary by itself — production still needs the
    child namespaced/no-egress — but it removes the in-process-RCE-in-the-CLI sink (#6).

    ``model_key`` (a validator/model fact, e.g. ``"MiniMax-M3"``) selects the per-model slot
    specialization (right activation reference + low-bit metric). None -> the generic slot.
    ``override_point`` (an override submission) composes the miner's epilogue into the
    validator-owned base kernel instead of loading a whole-kernel ``entry``."""
    from optima.sandbox import callable_from, load_module
    from optima.slots import slot_for_model

    slot = slot_for_model(slot_name, model_key)
    dtype = getattr(torch, dtype_name)
    direct_entry = None
    direct_op = None
    if bundle_path:
        from optima.rebuild import apply_rebuild_plan

        rebuild_phase = (
            "load"
            if os.environ.get("OPTIMA_PREBUILT_ARTIFACTS") == "1"
            else "all"
        )
        apply_rebuild_plan(bundle_path, phase=rebuild_phase)
        from optima.artifact_runtime import resolve_direct_artifact_entry
        from optima.manifest import load_manifest

        direct_op = load_manifest(bundle_path).op_for(slot_name, variant_name)
        if direct_op is None:
            raise ValueError(
                f"bundle has no manifest row for {(slot_name, variant_name)!r}"
            )
        direct_entry = resolve_direct_artifact_entry(direct_op)
    if direct_entry is not None:
        if override_point is not None:
            raise ValueError("direct CuTe AOT rows cannot use an override")
        entry = direct_entry
        prepare = _direct_aot_prepare_boundary(
            slot, direct_entry, prepare_name=prepare_name
        )
    else:
        # ONE module instance: entry/prepare (or an override's device fns) must share
        # a namespace — separate loads re-execute the candidate module body.
        module = load_module(source_path)  # candidate code; isolated child only
        if override_point is not None:
            from optima_kernels.override import build_override

            def _loader(name, _mod=module):
                fn = getattr(_mod, name, None)
                return fn if callable(fn) else None

            entry, prepare = build_override(
                slot_name, override_point, entry_name, _loader
            )
        else:
            entry = callable_from(module, entry_name)
            prepare = callable_from(module, prepare_name) if prepare_name else None
    eligibility = None
    if eligibility_metadata is not None:
        from optima.registry import eligibility_from_metadata

        eligibility = eligibility_from_metadata(
            eligibility_metadata, manifest_dtypes, manifest_architectures
        )
    return verify_entry(slot, entry, prepare=prepare, dtype=dtype, device=device, seed=seed,
                        shapes=shapes, jitter_seed=jitter_seed, graph_safe=graph_safe,
                        graph_replays=graph_replays, eligibility=eligibility,
                        tp_size=tp_size, world_size=world_size)


def _name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def format_verify(result: VerifyResult) -> str:
    graph = " graph=not-required"
    if result.graph_required:
        graph = " graph=verified" if result.graph_verified else " graph=NOT_VERIFIED"
    coverage = ""
    if result.coverage_required or result.num_not_applicable:
        coverage = (
            f" coverage={result.num_applicable}/{result.coverage_required}"
            f" n/a={result.num_not_applicable}"
        )
    if not result.domain_coverage_complete:
        coverage += " domain_coverage=INCOMPLETE"
        if result.domain_coverage_detail:
            coverage += f" ({result.domain_coverage_detail})"
    if result.context_inapplicable:
        status = "N/A"
    elif not result.passed:
        status = "FAIL"
    elif result.graph_required and not result.graph_verified:
        status = "NUMERICAL_PASS"
    else:
        status = "PASS"
    lines = [
        f"[{status}] {result.slot} "
        f"dtype={result.dtype}{graph}{coverage}"
    ]
    for r in result.shape_results:
        status = "N/A" if not r.applicable else ("ok " if r.passed else "FAIL")
        if r.metric == "cosine":
            score = f" cos={r.pass_ratio:.5f}"
        elif r.metric == "overlap":
            score = f" overlap={r.pass_ratio:.4f}"
        elif r.metric == "n/a":
            score = ""
        else:
            score = "" if r.pass_ratio >= 1.0 else f" ratio={r.pass_ratio:.4f}"
        replay = f" graph_replays={r.graph_replays}" if r.graph_replays else ""
        lines.append(
            f"  {status} shape={r.shape} max_abs={r.max_abs_err:.3e} max_rel={r.max_rel_err:.3e}{score}"
            + replay + (f"  {r.detail}" if r.detail else "")
        )
    return "\n".join(lines)
