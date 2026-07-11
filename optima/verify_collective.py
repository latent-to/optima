"""Distributed verification for collective slots (``kind="collective"``).

A collective (all-reduce, all-to-all, reduce-scatter) spans ranks, so it cannot be
verified on one GPU. This spawns ``world_size`` ranks, runs the miner's kernel as the
*real* collective on each, and compares every rank's output to the TRUSTED reference: a
``torch.distributed`` reduce of the **fp32** partials. Backend ``"gloo"`` runs a CPU
numeric check; ``"nccl"`` runs the real multi-GPU path. Both execute the dtype the
caller requested rather than silently substituting one.

The miner's kernel is handed the process group (the wider capability of a collective
slot); the validator owns the output buffer and the reference. This is the per-collective
gate — necessary but NOT sufficient: reduce error compounds across every layer, so the
end-to-end token/KL gate remains mandatory.

The strict JSON rank wire is a bounded parser boundary, not grading authority: miner
code shares each disposable rank process and can forge any in-process diagnostic.
The isolated executor and pristine reference arm own external qualification.
"""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from optima.capabilities import (
    CONTEXT_FIELDS,
    CallDescriptor,
    collective_call_descriptor,
)
from optima.registry import Eligibility
from optima.slots import SlotSpec
from optima.tensor_spec import allocate_output_spec, validate_output_allocation
from optima.verify import (
    _DEFAULT_GRAPH_REPLAYS,
    ShapeResult,
    VerifyResult,
    _compare_outputs,
    _clone_tensor_inputs,
    _CudaGraphBackend,
    _device_architecture,
    _graph_case_inputs,
    _input_bindings,
    _input_mutation_detail,
    _poison_outputs,
    _restore_tensor_inputs,
)


_VERDICT_VERSION = 1
_MAX_VERDICT_BYTES = 64 * 1024
_MAX_DETAIL_CHARS = 4096
_MAX_ERROR_CHARS = 16 * 1024
_VERDICT_FIELDS = frozenset({
    "version", "rank", "world_size", "passed", "score", "max_abs",
    "detail", "metric", "error", "graph_replays",
})
_VERDICT_METRICS = frozenset({"ratio", "cosine", "overlap"})


class CollectiveVerdictError(RuntimeError):
    pass


@dataclass(frozen=True)
class _RankVerdict:
    version: int
    rank: int
    world_size: int
    passed: bool
    score: float
    max_abs: float | None
    detail: str
    metric: str
    error: str | None
    graph_replays: int


def _number(value: Any, lo: float, hi: float | None = None, *, integer=False):
    valid_type = type(value) is int if integer else (
        isinstance(value, (int, float)) and not isinstance(value, bool)
    )
    number = value if integer else float(value) if valid_type else float("nan")
    if (not valid_type or not math.isfinite(number) or number < lo
            or (hi is not None and number > hi)):
        raise CollectiveVerdictError("numeric field is outside its permitted range")
    return number


def _regular_identity(path: Path) -> tuple[int, int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise CollectiveVerdictError("verdict destination is not a regular file")
    return int(info.st_dev), int(info.st_ino)


def _bound_fd(path: Path, flags: int, identity: tuple[int, int]):
    fd = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0))
    info = os.fstat(fd)
    if (not stat.S_ISREG(info.st_mode)
            or (int(info.st_dev), int(info.st_ino)) != identity):
        os.close(fd)
        raise CollectiveVerdictError("verdict destination inode changed or is not regular")
    return fd, info


def _write_rank_verdict(
    path: Path, verdict: _RankVerdict, expected_identity: tuple[int, int]
) -> None:
    body = json.dumps(
        asdict(verdict), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if not body or len(body) > _MAX_VERDICT_BYTES:
        raise CollectiveVerdictError("rank verdict exceeds the wire-size limit")
    fd, _ = _bound_fd(path, os.O_WRONLY | os.O_TRUNC, expected_identity)
    with os.fdopen(fd, "wb") as stream:
        if stream.write(body) != len(body):
            raise CollectiveVerdictError("short verdict write")
        stream.flush()
        os.fsync(stream.fileno())


def _reject_constant(value: str) -> None:
    raise CollectiveVerdictError(f"non-finite JSON constant {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len({key for key, _ in pairs}) != len(pairs):
        raise CollectiveVerdictError("duplicate JSON key")
    return dict(pairs)


def _read_rank_verdict(
    path: Path,
    *,
    expected_rank: int,
    expected_world_size: int,
    expected_identity: tuple[int, int],
) -> _RankVerdict:
    if _regular_identity(path) != expected_identity:
        raise CollectiveVerdictError("verdict destination inode changed")
    fd, info = _bound_fd(path, os.O_RDONLY, expected_identity)
    size = int(info.st_size)
    if not 1 <= size <= _MAX_VERDICT_BYTES:
        os.close(fd)
        raise CollectiveVerdictError(f"verdict size {size} is outside the limit")
    try:
        with os.fdopen(fd, "rb") as stream:
            raw = stream.read(_MAX_VERDICT_BYTES + 1)
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except (CollectiveVerdictError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectiveVerdictError(f"invalid verdict JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != _VERDICT_FIELDS:
        raise CollectiveVerdictError("verdict fields do not match the exact schema")
    version = _number(value["version"], 1, 1, integer=True)
    rank = _number(value["rank"], 0, expected_world_size - 1, integer=True)
    world_size = _number(value["world_size"], 1, integer=True)
    if rank != expected_rank or world_size != expected_world_size:
        raise CollectiveVerdictError("rank identity mismatch")
    passed = value["passed"]
    if type(passed) is not bool:
        raise CollectiveVerdictError("passed must be a boolean")
    score = _number(value["score"], -1.0, 1.0)
    max_abs = None if value["max_abs"] is None else _number(value["max_abs"], 0.0)
    detail = value["detail"]
    if not isinstance(detail, str) or len(detail) > _MAX_DETAIL_CHARS:
        raise CollectiveVerdictError("detail must be a bounded string")
    metric = value["metric"]
    if metric not in _VERDICT_METRICS:
        raise CollectiveVerdictError(f"unsupported verdict metric {metric!r}")
    error = value["error"]
    if error is not None and (
        not isinstance(error, str) or len(error) > _MAX_ERROR_CHARS
    ):
        raise CollectiveVerdictError("error must be null or a bounded string")
    graph_replays = _number(value["graph_replays"], 0, 64, integer=True)
    if passed and (error is not None or max_abs is None):
        raise CollectiveVerdictError("passing verdict has missing output or an error")
    return _RankVerdict(version, rank, world_size, passed, score, max_abs, detail,
                        metric, error, graph_replays)


def _collective_descriptor(
    shape: dict,
    *,
    slot_name: str | None = None,
    dtype_name: str,
    device: str,
    graph_safe: bool,
    model_key: str | None,
    architecture: str | None,
    world_size: int,
) -> CallDescriptor:
    dimensions: dict[str, Any] = {}
    if slot_name in {
        "moe.fused_experts_reduce",
        "collective.moe_finalize_ar_rmsnorm",
    }:
        # These two contracts explicitly exclude expert parallel dispatch/combine.
        # EP is not a semantic dimension of a bare all-reduce or AR+norm call.
        dimensions["ep_size"] = 1
    aliases = {
        "alignment": "alignment", "batch_size": "batch_size", "batch": "batch_size",
        "block_size": "block_size", "head_dim": "head_dim", "kv_len": "kv_len",
        "exp_tokens": "exp_tokens", "num_experts": "num_experts",
        "num_kv_heads": "num_kv_heads", "num_q_heads": "num_q_heads",
        "num_tokens": "num_tokens", "page_size": "page_size", "q_len": "q_len",
        "top_k": "top_k", "topk": "top_k", "inter": "intermediate_dim",
    }
    dimensions.update(
        {dst: shape[src] for src, dst in aliases.items() if src in shape}
    )
    if "hidden" in shape:
        dimensions.update(hidden_dim=shape["hidden"], last_dim=shape["hidden"])
    return collective_call_descriptor(
        dtype=dtype_name,
        architecture=architecture or _device_architecture(device),
        graph_mode="cuda_graph" if graph_safe and device == "cuda" else "eager",
        # Live Sglang collective bindings do not yet receive a canonical arena-model
        # identity. Omitting it here makes model-constrained variants consistently
        # N/A instead of qualifying offline and then never routing live. PR3 manifests
        # will supply the trusted identity to both sides together.
        model=None,
        world_size=world_size,
        dimensions=dimensions,
    )


def _match_detail(match) -> str:
    return "; ".join(
        f"{m.field} {m.reason}: expected {m.expected}"
        + ("" if m.actual is None else f", got {m.actual!r}")
        for m in match.mismatches
    )


def _terminate_processes(processes, *, grace_s: float = 5.0) -> None:
    import signal
    import time

    for sig in (signal.SIGTERM, signal.SIGKILL):
        for process in processes:
            if not process.is_alive() or process.pid is None:
                continue
            try:
                if os.getpgid(process.pid) == process.pid:
                    os.killpg(process.pid, sig)
                elif sig == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()
            except OSError:
                process.terminate() if sig == signal.SIGTERM else process.kill()
        deadline = time.monotonic() + grace_s
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))


def _rank_worker(rank, world_size, backend, init_method, slot_name, source_path, entry_name,
                 shape, dtype_name, device, seed, result_dir, prepare_name=None, model_key=None,
                 bundle_path=None, graph_safe=False,
                 graph_replays=_DEFAULT_GRAPH_REPLAYS, verdict_identities=(),
                 run_mode="single"):
    """One rank: init the group, run the miner collective into a validator-owned buffer,
    compare to the trusted fp32 cross-rank reduce. Writes its verdict to ``result_dir``.

    Slot-driven via the slot's ``collective_partial`` (the fp32 tensor whose cross-rank
    SUM is the reference) and ``invoke_collective`` (how to call the kernel with the group),
    so this handles a bare all-reduce AND a block that owns its trailing reduce
    (moe.fused_experts_reduce) without hard-coding either contract.

    ``run_mode`` separates three distinct claims: ``single`` is one clean-room call;
    ``temporal_eager`` is an unsynchronized multi-call burst that exposes protocol
    state; ``graph_sequence`` captures multiple shapes in one loaded module and catches
    first-shape/workspace caches. Conflating the latter two would synchronize away the
    rank-skew condition the temporal gate exists to test.
    """
    try:
        import ctypes
        import signal

        os.setsid()
        # Linux: a killed parent must not strand NCCL workers or their children.
        ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGKILL)
    except Exception:  # noqa: BLE001 - non-Linux dev hosts keep bounded parent cleanup
        pass

    import torch
    import torch.distributed as dist

    verdict = {
        "version": _VERDICT_VERSION,
        "rank": rank,
        "world_size": world_size,
        "passed": False,
        "score": 0.0,
        "max_abs": None,
        "detail": "",
        "metric": "ratio",
        "error": None,
        "graph_replays": 0,
    }
    initialized = False
    graph_capture_attempted = False
    try:
        from optima.sandbox import callable_from, load_module
        from optima.slots import slot_for_model

        if device == "cuda":
            torch.cuda.set_device(rank)
        dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
        initialized = True

        # A bundle with a rebuild plan (e.g. declared cuda_sources compiled by a reviewed
        # patcher) needs that plan applied in EVERY process that loads the kernel — these
        # spawned ranks included, or the shim silently falls back to its reference path
        # and the "verify" validates nothing (phantom parity). Rank 0 builds (compile is
        # not concurrency-safe on the shared cache), the rest barrier then load from cache.
        if bundle_path:
            from optima.rebuild import apply_rebuild_plan

            if rank == 0:
                apply_rebuild_plan(bundle_path)
            dist.barrier()
            if rank != 0:
                apply_rebuild_plan(bundle_path)

        slot = slot_for_model(slot_name, model_key)
        dtype = getattr(torch, dtype_name)
        dev = f"cuda:{rank}" if device == "cuda" else "cpu"

        # ONE module instance for prepare+entry AND across every sequence step (separate
        # loads would re-execute the body and split namespaces; and the whole point of a
        # temporal sequence is that the kernel's cross-call state PERSISTS step to step).
        module = load_module(source_path)
        entry = callable_from(module, entry_name)
        prepare_fn = callable_from(module, prepare_name) if prepare_name else None
        invoke = slot.invoke_collective or (lambda e, i, o, g, p: e(i["x"], o, g))

        steps = list(shape) if isinstance(shape, (list, tuple)) else [shape]
        if run_mode not in {"single", "temporal_eager", "graph_sequence"}:
            raise ValueError(f"unknown collective verify run mode {run_mode!r}")
        if run_mode == "single" and len(steps) != 1:
            raise ValueError("single collective verify mode requires exactly one shape")
        if run_mode != "single" and len(steps) < 2:
            raise ValueError(f"{run_mode} requires at least two shapes")
        tol = slot.tolerance_for(dtype)

        # PHASE 1 — launch every step's kernel BACK-TO-BACK with no intervening
        # collective/sync. A sequence must reproduce the ENGINE's concurrency regime:
        # per-layer calls stream one after another and ranks skew freely (a rank that
        # finishes call N launches N+1 while a peer still consumes N — the case the
        # kernel's buffer rotation must tolerate). Interleaving the trusted reference's
        # all_reduce here would force cross-rank lockstep after every call and hide
        # exactly that class. Inputs/outputs stay alive for phase 2.
        staged = []
        # The temporal sequence intentionally remains an unsynchronized eager
        # burst. The graph sequence is a separate worker so capture synchronization
        # cannot erase rank skew from this protocol-state check.
        verify_graph = bool(
            graph_safe
            and device == "cuda"
            and run_mode in {"single", "graph_sequence"}
        )
        # Live layers prepare invariant weights once, then reuse the same prepared
        # object across token buckets/calls. Cache by validator-observed static input
        # ABI and retain the first static tensor objects for every repeated key.
        prepared_cache: dict[tuple, tuple[Any, dict[str, Any]]] = {}
        for si, step in enumerate(steps):
            # Fresh data per step (a step must not be able to pass by leaving its
            # PREVIOUS output in place), same shapes on every rank.
            inputs = slot.make_inputs(dtype=dtype, device=dev, seed=seed + 7919 * si,
                                      rank=rank, world_size=world_size, **step)
            prepared = None
            prepare_key = None
            static_inputs: dict[str, Any] = {}
            if prepare_fn is not None and slot.invoke_prepare is not None:
                dynamic = set(slot.graph_dynamic_inputs)
                static_inputs = {
                    name: value for name, value in inputs.items() if name not in dynamic
                }
                prepare_key = tuple(
                    (
                        name,
                        "tensor",
                        tuple(value.shape),
                        str(value.dtype),
                    )
                    if torch.is_tensor(value)
                    else (name, "scalar", type(value).__name__, repr(value))
                    for name, value in sorted(static_inputs.items())
                )
                cached = prepared_cache.get(prepare_key)
                if cached is not None:
                    prepared, retained_static = cached
                    inputs.update(retained_static)
            # Candidate code never receives these storages. References are derived
            # from them in phase 2, even if prepare/entry corrupt the live tensors.
            trusted_inputs = _clone_tensor_inputs(inputs)
            input_bindings = _input_bindings(inputs)

            replay_inputs = []
            if verify_graph:
                for replay in range(graph_replays):
                    last_error = ""
                    for attempt in range(8):
                        fresh = slot.make_inputs(
                            dtype=dtype,
                            device=dev,
                            seed=(
                                seed + 7919 * si + 104_729 * (replay + 1)
                                + 1_000_003 * attempt
                            ),
                            rank=rank,
                            world_size=world_size,
                            **step,
                        )
                        try:
                            logical = _graph_case_inputs(
                                slot, trusted_inputs, fresh
                            )
                        except RuntimeError as exc:
                            last_error = str(exc)
                            continue
                        replay_inputs.append(logical)
                        break
                    else:
                        raise RuntimeError(
                            last_error or "could not generate fresh graph inputs"
                        )

            # (prepare, forward) collective blocks (e.g. moe.fused_experts_reduce): run
            # the miner's weight-prep on THIS rank's shard before the forward.
            if (
                prepare_fn is not None
                and slot.invoke_prepare is not None
                and prepare_key not in prepared_cache
            ):
                prepared = slot.invoke_prepare(prepare_fn, inputs)
                mutation = _input_mutation_detail(
                    inputs, trusted_inputs, input_bindings
                )
                if mutation:
                    raise RuntimeError(f"prepare {mutation}")
                prepared_cache[prepare_key] = (prepared, static_inputs)

            # Validator-owned output buffer(s). Single-output slots keep the original
            # tensor-valued call shape; multi-output slots (e.g. ar_residual_rmsnorm's
            # [norm_out, new_residual]) receive the list.
            output_contract = slot.output_contract(inputs)
            allocation = allocate_output_spec(
                output_contract,
                fallback_dtype=dtype,
                fallback_device=dev,
                inputs=(v for v in inputs.values() if torch.is_tensor(v)),
            )
            outs = allocation.outputs
            out_arg = outs[0] if len(outs) == 1 else outs

            invoke(entry, inputs, out_arg, dist.group.WORLD, prepared)  # miner fills the buffers
            def validate_outputs():
                validate_output_allocation(
                    output_contract,
                    allocation,
                    fallback_dtype=dtype,
                    fallback_device=dev,
                    inputs=(
                        value for value in inputs.values() if torch.is_tensor(value)
                    ),
                )

            validate_outputs()

            if verify_graph:
                # Snapshot eager output before warmup/capture overwrite the same
                # validator-owned buffers. The clone is ordered after the candidate
                # on the same stream.
                output_sets = [("eager", [out.detach().clone() for out in outs])]
                graph_capture_attempted = True
                graph_backend = _CudaGraphBackend(outs[0].device)

                def graph_invoke():
                    invoke(entry, inputs, out_arg, dist.group.WORLD, prepared)

                try:
                    graph_backend.synchronize()
                    mutation = _input_mutation_detail(
                        inputs, trusted_inputs, input_bindings
                    )
                    if mutation:
                        raise RuntimeError(mutation)
                    _restore_tensor_inputs(inputs, trusted_inputs, input_bindings)
                    graph_backend.warmup(graph_invoke)
                    graph_backend.synchronize()
                    validate_outputs()
                    mutation = _input_mutation_detail(
                        inputs, trusted_inputs, input_bindings
                    )
                    if mutation:
                        raise RuntimeError(mutation)
                    _restore_tensor_inputs(inputs, trusted_inputs, input_bindings)
                    graph = graph_backend.capture(graph_invoke)
                    graph_backend.synchronize()
                    validate_outputs()
                    mutation = _input_mutation_detail(
                        inputs, trusted_inputs, input_bindings
                    )
                    if mutation:
                        raise RuntimeError(mutation)
                except Exception as exc:  # noqa: BLE001 - false graph_safe claim
                    raise RuntimeError(
                        f"cuda graph capture failed for {slot.name}: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                checked_sets = [
                    ("eager", output_sets[0][1], trusted_inputs)
                ]
                for replay, logical in enumerate(replay_inputs):
                    try:
                        _restore_tensor_inputs(inputs, logical, input_bindings)
                        _poison_outputs(outs, replay)
                        graph_backend.synchronize()
                        graph_backend.replay(graph)
                        graph_backend.synchronize()
                        validate_outputs()
                        mutation = _input_mutation_detail(
                            inputs, logical, input_bindings
                        )
                        if mutation:
                            raise RuntimeError(mutation)
                    except Exception as exc:  # noqa: BLE001 - every replay is required
                        raise RuntimeError(
                            f"cuda graph replay[{replay}] failed for {slot.name}: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    checked_sets.append(
                        (
                            f"cuda graph replay[{replay}]",
                            [out.detach().clone() for out in outs],
                            logical,
                        )
                    )
                output_sets = checked_sets
                verdict["graph_replays"] = graph_replays
            else:
                output_sets = [("eager", outs, trusted_inputs)]

            # Retain the complete allocation so future declared workspaces cannot
            # be reclaimed before comparison/replay finishes.
            staged.append(
                (
                    si,
                    step,
                    inputs,
                    trusted_inputs,
                    input_bindings,
                    allocation,
                    output_sets,
                )
            )

        # PHASE 2 — trusted references + comparison, after the whole burst: the fp32
        # cross-rank SUM of each rank's partial, then the slot's trusted post-reduce
        # math (collective_finish) if it does local work after the sum (residual add /
        # norm). No finish -> the sum IS the single expected output (bare all-reduce).
        passed, max_abs, score, detail, metric = True, 0.0, 1.0, "", "ratio"
        for (
            si,
            step,
            inputs,
            trusted_inputs,
            input_bindings,
            _allocation,
            output_sets,
        ) in staged:
            if not verify_graph:
                mutation = _input_mutation_detail(
                    inputs, trusted_inputs, input_bindings
                )
                if mutation:
                    raise RuntimeError(mutation)
            for output_label, checked_outs, reference_inputs in output_sets:
                # Trusted local math never consumes miner-prepared state. It runs
                # only on the pre-candidate snapshot, then the validator-owned group
                # performs the fp32 cross-rank reference reduce.
                partial = (
                    slot.collective_partial(reference_inputs, None)
                    if slot.collective_partial
                    else reference_inputs["x"].float()
                )
                summed = partial.detach().float().clone()
                dist.all_reduce(summed, op=dist.ReduceOp.SUM)
                refs = (
                    slot.collective_finish(reference_inputs, summed, None)
                    if slot.collective_finish is not None
                    else [summed]
                )
                current = _compare_outputs(
                    checked_outs, list(refs), tol=tol, correctness=slot.correctness
                )
                passed = passed and current.passed
                max_abs = max(max_abs, current.max_abs)
                score = min(score, current.min_score)
                metric = current.metric
                if not current.passed and not detail:
                    prefix = f"step {si} {step}: " if len(steps) > 1 else ""
                    detail = (
                        f"{prefix}{output_label}: "
                        f"{current.detail or 'output mismatch'}"
                    )
        verdict.update(passed=passed, score=score, max_abs=max_abs,
                       detail=detail, metric=metric)
    except BaseException:  # noqa: BLE001 - report any failure as a fail
        import traceback
        verdict["error"] = traceback.format_exc()[-_MAX_ERROR_CHARS:]
    finally:
        if verdict["max_abs"] is not None and not math.isfinite(verdict["max_abs"]):
            verdict["max_abs"] = None
        try:
            _write_rank_verdict(
                Path(result_dir) / f"rank{rank}.json",
                _RankVerdict(**verdict),
                tuple(verdict_identities[rank]),
            )
        except Exception:  # noqa: BLE001
            pass
        if initialized and device == "cuda" and graph_capture_attempted:
            # torch/NCCL can hang while destroying a process group that owns a
            # captured collective. These are disposable workers; the parent grades
            # the durable strict verdict, and successful process exit releases the
            # CUDA context without trusting destructor progress.
            os._exit(0)
        try:
            import torch.distributed as dist
            if initialized and dist.is_initialized():
                # Never add a final barrier: one failed rank would turn cleanup into
                # an unbounded collective. Eager/gloo groups destroy independently.
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass


def verify_collective(
    slot: SlotSpec,
    source_path: str,
    entry_name: str,
    *,
    prepare_name: str | None = None,
    world_size: int = 2,
    backend: str | None = None,
    device: str | None = None,
    dtype_name: str | None = None,
    seed: int = 0,
    shapes: list[dict] | None = None,
    model_key: str | None = None,
    jitter_seed: int | None = None,
    bundle_path: str | None = None,
    graph_safe: bool = False,
    graph_replays: int = _DEFAULT_GRAPH_REPLAYS,
    timeout_s: float | None = None,
    eligibility: Eligibility | None = None,
    tp_size: int | None = None,
) -> VerifyResult:
    """Verify a collective slot's kernel across ``world_size`` spawned ranks.

    Defaults: ``device`` = cuda iff enough GPUs, else cpu; ``backend`` = nccl on cuda,
    gloo on cpu; dtype = bf16 on CUDA and fp32 on CPU. An explicit dtype is executed
    exactly, never replaced by a proxy dtype. ``model_key`` selects
    the validator per-model slot profile (activation reference + metric); None -> generic.
    ``jitter_seed`` perturbs the count dims per run (same anti-shape-branching guard as
    the per-op verify — without it a collective kernel could hard-code the fixed verify
    shapes); jittered in the parent so every rank builds identical shapes. A declared
    graph-safe CUDA collective is captured once per applicable clean-room shape and all
    outputs are poisoned and graded after every replay. CPU/gloo remains numerical-only.
    Eligibility is resolved in the trusted parent before rebuild or miner import, so an
    off-architecture variant is N/A rather than an attempted compile or phantom pass.
    """
    import torch
    import torch.multiprocessing as mp
    import time

    from optima.verify import _jitter_shapes

    if slot.kind != "collective":
        raise ValueError(f"slot {slot.name!r} is not a collective")
    if type(world_size) is not int or world_size < 2:
        raise ValueError("distributed collective verification requires world_size >= 2")
    if tp_size is not None and (type(tp_size) is not int or tp_size != world_size):
        raise ValueError(
            "collective tp_size must equal the actual WORLD process-group size"
        )
    if device is None:
        device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() >= world_size) else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("collective device must be 'cpu' or 'cuda'")
    if device == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.device_count() < world_size
    ):
        raise ValueError(
            f"collective CUDA verify needs {world_size} visible GPUs"
        )
    if backend is None:
        backend = "nccl" if device == "cuda" else "gloo"
    if backend not in {"gloo", "nccl"}:
        raise ValueError("collective backend must be 'gloo' or 'nccl'")
    if (device, backend) not in {("cpu", "gloo"), ("cuda", "nccl")}:
        raise ValueError("collective verify requires cpu/gloo or cuda/nccl")
    if graph_safe and graph_replays < 2:
        raise ValueError("CUDA graph verification requires at least two replays")
    if timeout_s is None:
        timeout_s = float(os.environ.get("OPTIMA_COLLECTIVE_VERIFY_TIMEOUT_S", "900"))
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("collective verify timeout must be a finite positive number")
    if dtype_name is None:
        dtype_name = "float32" if device == "cpu" else "bfloat16"
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype) or not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError(f"collective dtype {dtype_name!r} is not a floating torch dtype")
    architecture = None
    if device == "cuda":
        capabilities = {
            tuple(torch.cuda.get_device_capability(rank))
            for rank in range(world_size)
        }
        if len(capabilities) != 1:
            raise ValueError(
                "collective CUDA verify requires homogeneous device architectures"
            )
        major, minor = capabilities.pop()
        architecture = f"sm{major}{minor}"
    catalog_shapes = [dict(shape) for shape in (shapes if shapes is not None else slot.shapes)]
    test_shapes = list(catalog_shapes)
    if jitter_seed is not None:
        test_shapes = _jitter_shapes(catalog_shapes, jitter_seed)

    def _spawn_and_collect(
        shape_or_seq,
        label: dict,
        run_seed: int,
        *,
        run_mode: str = "single",
    ) -> ShapeResult:
        with tempfile.TemporaryDirectory(prefix="optima_collective_") as rd:
            init_method = f"file://{os.path.join(rd, 'pg_store')}"
            verdict_identities: list[tuple[int, int]] = []
            for rank in range(world_size):
                path = Path(rd) / f"rank{rank}.json"
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                os.close(fd)
                verdict_identities.append(_regular_identity(path))
            args = (world_size, backend, init_method, slot.name, source_path, entry_name,
                    shape_or_seq, dtype_name, device, run_seed, rd, prepare_name, model_key,
                    bundle_path, graph_safe, graph_replays, verdict_identities, run_mode)
            spawn_err: str | None = None
            processes = []
            try:
                context = mp.spawn(
                    _rank_worker, args=args, nprocs=world_size, join=False
                )
                processes = list(context.processes)
                deadline = time.monotonic() + timeout_s
                joined = False
                while time.monotonic() < deadline:
                    remaining = max(0.0, deadline - time.monotonic())
                    if context.join(timeout=min(1.0, remaining)):
                        joined = True
                        break
                if not joined:
                    spawn_err = f"collective verify timed out after {timeout_s:g}s"
                    _terminate_processes(processes)
            except Exception as exc:  # noqa: BLE001 - a crashed rank; verdicts still on disk
                spawn_err = f"collective worker join failed: {type(exc).__name__}: {exc}"
                _terminate_processes(processes)
            finally:
                alive = [process for process in processes if process.is_alive()]
                if alive:
                    _terminate_processes(alive)

            exitcodes = [process.exitcode for process in processes]
            if spawn_err is None and (
                len(exitcodes) != world_size
                or any(code != 0 for code in exitcodes)
            ):
                spawn_err = f"collective worker exit codes were {exitcodes}"

            verdicts: list[_RankVerdict] = []
            verdict_errors: list[str] = []
            for rank, identity in enumerate(verdict_identities):
                path = Path(rd) / f"rank{rank}.json"
                try:
                    verdicts.append(
                        _read_rank_verdict(
                            path,
                            expected_rank=rank,
                            expected_world_size=world_size,
                            expected_identity=identity,
                        )
                    )
                except (OSError, CollectiveVerdictError) as exc:
                    verdict_errors.append(f"rank {rank}: {exc}")

        if not verdicts:
            wire_detail = "; ".join(verdict_errors) if verdict_errors else "none"
            return ShapeResult(shape=label, dtype=dtype_name, passed=False,
                               max_abs_err=float("inf"), max_rel_err=float("inf"),
                               pass_ratio=0.0,
                               detail=(f"no valid rank verdicts (spawn={spawn_err}; "
                                       f"wire={wire_detail})"))
        rank_errs = [verdict for verdict in verdicts if verdict.error]
        passed = (
            spawn_err is None
            and len(verdicts) == world_size
            and not verdict_errors
            and not rank_errs
            and all(verdict.passed for verdict in verdicts)
        )
        worst = min(verdicts, key=lambda verdict: verdict.score)
        detail = ""
        if spawn_err is not None:
            detail = spawn_err
        elif verdict_errors:
            detail = "invalid rank verdict: " + "; ".join(verdict_errors)
        elif rank_errs:
            detail = (
                f"rank {rank_errs[0].rank} raised: "
                + rank_errs[0].error.strip().splitlines()[-1]
            )
        elif len(verdicts) != world_size:
            detail = (
                f"missing rank verdicts: got {[v.rank for v in verdicts]}, "
                f"expected {list(range(world_size))}"
            )
        elif not passed:
            detail = (
                f"worst rank {worst.rank}: {worst.metric}={worst.score:.4f} "
                f"{worst.detail}"
            )
        return ShapeResult(
            shape=label, dtype=dtype_name, passed=passed,
            max_abs_err=max(
                float("inf") if verdict.max_abs is None else verdict.max_abs
                for verdict in verdicts
            ),
            max_rel_err=0.0,
            pass_ratio=worst.score,
            detail=detail,
            metric=worst.metric,
            graph_replays=min(verdict.graph_replays for verdict in verdicts),
        )

    results: list[ShapeResult] = []
    applicable_shapes: list[dict] = []
    applicable_single_results: list[ShapeResult] = []
    context_blocked: list[bool] = []
    static_context_fields = CONTEXT_FIELDS | {"ep_size", "tp_size", "world_size"}
    for i, (catalog_shape, jittered_shape) in enumerate(
        zip(catalog_shapes, test_shapes)
    ):
        shape = jittered_shape
        if (
            eligibility is not None
            and slot.name == "collective.moe_finalize_ar_rmsnorm"
            and dtype_name != "bfloat16"
        ):
            # The live FlashInfer export ABI is a 16-bit BF16 pointer contract.
            # CPU float32 remains useful for validator-owned reference tests when no
            # bundle eligibility is being qualified, but a float32 miner variant must
            # be N/A rather than PASS offline and remain unreachable live.
            context_blocked.append(True)
            results.append(
                ShapeResult(
                    shape=shape,
                    dtype=dtype_name,
                    passed=True,
                    max_abs_err=0.0,
                    max_rel_err=0.0,
                    pass_ratio=1.0,
                    detail=(
                        "validator N/A (outside live deep-export dtype domain): "
                        "requires bfloat16"
                    ),
                    metric="n/a",
                    applicable=False,
                )
            )
            continue
        if eligibility is not None:
            candidates = [shape] + ([] if shape == catalog_shape else [catalog_shape])
            for candidate in candidates:
                match = eligibility.match(
                    _collective_descriptor(
                        candidate, slot_name=slot.name,
                        dtype_name=dtype_name, device=device,
                        graph_safe=graph_safe, model_key=model_key,
                        architecture=architecture,
                        world_size=world_size,
                    )
                )
                if match.accepted:
                    shape = candidate
                    break
            if not match.accepted:
                context_blocked.append(
                    bool(match.mismatches)
                    and any(
                        mismatch.field in static_context_fields
                        for mismatch in match.mismatches
                    )
                )
                results.append(
                    ShapeResult(
                        shape=shape,
                        dtype=dtype_name,
                        passed=True,
                        max_abs_err=0.0,
                        max_rel_err=0.0,
                        detail=(
                            "validator N/A (outside declared capability domain): "
                            + _match_detail(match)
                        ),
                        metric="n/a",
                        applicable=False,
                    )
                )
                continue
        context_blocked.append(False)
        current = _spawn_and_collect(shape, shape, seed + i)
        results.append(current)
        applicable_shapes.append(shape)
        applicable_single_results.append(current)

    # TEMPORAL sequence gate: the per-shape spawns above are clean-room single calls —
    # structurally blind to cross-call protocol state (IPC workspaces, per-token
    # counters, rotating sentinel buffers). Replay the same shapes big -> small -> big
    # inside ONE process: a kernel whose clears/rotation only maintain invariants for
    # the current call's rows is exactly right on every fresh call and corrupts the
    # first time the token count GROWS between calls (the 2026-07-07 engine failure;
    # ascending order would never trip it, grow-back does).
    seq = _temporal_sequence(applicable_shapes)
    if seq is not None:
        base = seq[: len(seq) // 3]
        label = {"sequence": "burst x3: num_tokens "
                             + "->".join(str(s.get("num_tokens")) for s in base)}
        results.append(
            _spawn_and_collect(
                seq, label, seed + 10_007, run_mode="temporal_eager"
            )
        )

    graph_sequence_result = None
    graph_seq = _graph_capture_sequence(applicable_shapes)
    if graph_safe and device == "cuda" and graph_seq is not None:
        label = {
            "sequence": "same-process cuda graphs: num_tokens "
            + "->".join(str(shape.get("num_tokens")) for shape in graph_seq)
        }
        graph_sequence_result = _spawn_and_collect(
            graph_seq,
            label,
            seed + 20_011,
            run_mode="graph_sequence",
        )
        results.append(graph_sequence_result)

    applicable_results = [result for result in results if result.applicable]
    coverage_required = 1 if eligibility is not None else 0
    coverage_sufficient = (
        len(applicable_results) >= coverage_required
        if coverage_required
        else bool(applicable_results)
    )
    graph_verified = bool(
        graph_safe
        and device == "cuda"
        and applicable_single_results
        and all(
            result.passed and result.graph_replays == graph_replays
            for result in (
                applicable_single_results
                + ([] if graph_sequence_result is None else [graph_sequence_result])
            )
        )
    )
    context_inapplicable = bool(
        eligibility is not None
        and not applicable_results
        and context_blocked
        and all(context_blocked)
    )
    return VerifyResult(
        slot=slot.name,
        dtype=dtype_name,
        passed=coverage_sufficient and all(r.passed for r in applicable_results),
        shape_results=results,
        graph_required=bool(graph_safe),
        graph_verified=graph_verified,
        coverage_required=coverage_required,
        context_inapplicable=context_inapplicable,
    )


def _temporal_sequence(shapes: list[dict]) -> list[dict] | None:
    """Adversarial call ORDER over the (already jittered) shapes: descending token
    counts then back up (343 -> 87 -> 45 -> 9 -> 45 -> 87 -> 343), TILED — enough
    back-to-back calls to cycle every comm-buffer generation several times at every
    token range, with mode transitions (one-shot/two-shot bands) in both directions.
    Combined with the rank worker's no-sync burst launch this approximates the engine
    regime (per-layer streaming calls, rank skew) that clean-room single calls can
    never see. None when the slot has no num_tokens dimension or fewer than two
    distinct counts (nothing temporal to vary)."""
    keyed = [(s.get("num_tokens"), s) for s in shapes]
    if any(k is None for k, _ in keyed) or len({k for k, _ in keyed}) < 2:
        return None
    desc = [s for _, s in sorted(keyed, key=lambda t: t[0], reverse=True)]
    return (desc + desc[-2::-1]) * 3


def _graph_capture_sequence(shapes: list[dict]) -> list[dict] | None:
    """Exercise multiple captured shapes in one module/process.

    Singleton workers prove each graph in clean state. This grow-then-shrink sequence
    separately catches kernels that cache the first captured shape or workspace and
    therefore fail only when an engine captures another bucket in the same process.
    """

    if len(shapes) < 2:
        return None
    if all(shape.get("num_tokens") is not None for shape in shapes):
        ordered = sorted(shapes, key=lambda shape: shape["num_tokens"])
    else:
        ordered = list(shapes)
    return ordered + ordered[-2::-1]
