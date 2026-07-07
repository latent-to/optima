"""Distributed verification for collective slots (``kind="collective"``).

A collective (all-reduce, all-to-all, reduce-scatter) spans ranks, so it cannot be
verified on one GPU. This spawns ``world_size`` ranks, runs the miner's kernel as the
*real* collective on each, and compares every rank's output to the TRUSTED reference: a
``torch.distributed`` reduce of the **fp32** partials. Backend ``"gloo"`` runs on CPU (a
numeric check, no GPU — note gloo has no bf16, so the CPU path uses fp32); ``"nccl"``
runs the real multi-GPU path.

The miner's kernel is handed the process group (the wider capability of a collective
slot); the validator owns the output buffer and the reference. This is the per-collective
gate — necessary but NOT sufficient: reduce error compounds across every layer, so the
end-to-end token/KL gate remains mandatory.
"""

from __future__ import annotations

import os
import pickle
import tempfile

from optima.slots import SlotSpec
from optima.verify import ShapeResult, VerifyResult, _compare, _name

_DTYPES = {"float32": "float32", "bfloat16": "bfloat16", "float16": "float16"}


def _rank_worker(rank, world_size, backend, init_method, slot_name, source_path, entry_name,
                 shape, dtype_name, device, seed, result_dir, prepare_name=None, model_key=None,
                 bundle_path=None):
    """One rank: init the group, run the miner collective into a validator-owned buffer,
    compare to the trusted fp32 cross-rank reduce. Writes its verdict to ``result_dir``.

    Slot-driven via the slot's ``collective_partial`` (the fp32 tensor whose cross-rank
    SUM is the reference) and ``invoke_collective`` (how to call the kernel with the group),
    so this handles a bare all-reduce AND a block that owns its trailing reduce
    (moe.fused_experts_reduce) without hard-coding either contract.

    ``shape`` is one shape dict (single clean-room call — the per-shape gate) OR a LIST
    of shape dicts: a TEMPORAL sequence run back-to-back in THIS process against the
    same loaded kernel, checking every step. Comm-heavy kernels keep cross-call protocol
    state (IPC workspaces, per-token counters, rotating sentinel buffers); a kernel can
    be exactly right on any fresh-state single call and corrupt the first time the token
    count GROWS between calls (caught in-engine 2026-07-07: stale Lamport payloads read
    as arrived). Only a multi-call sequence in one process can see that class."""
    import torch
    import torch.distributed as dist

    verdict = {"rank": rank, "passed": False, "score": 0.0, "max_abs": float("inf"),
               "detail": "", "metric": "ratio", "err": None}
    initialized = False
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
        invoke = slot.invoke_collective or (lambda e, i, o, g, p: e(i["x"], o, g))

        steps = list(shape) if isinstance(shape, (list, tuple)) else [shape]
        tol = slot.tolerance_for(dtype)

        # PHASE 1 — launch every step's kernel BACK-TO-BACK with no intervening
        # collective/sync. A sequence must reproduce the ENGINE's concurrency regime:
        # per-layer calls stream one after another and ranks skew freely (a rank that
        # finishes call N launches N+1 while a peer still consumes N — the case the
        # kernel's buffer rotation must tolerate). Interleaving the trusted reference's
        # all_reduce here would force cross-rank lockstep after every call and hide
        # exactly that class. Inputs/outputs stay alive for phase 2.
        staged = []
        for si, step in enumerate(steps):
            # Fresh data per step (a step must not be able to pass by leaving its
            # PREVIOUS output in place), same shapes on every rank.
            inputs = slot.make_inputs(dtype=dtype, device=dev, seed=seed + 7919 * si,
                                      rank=rank, world_size=world_size, **step)

            # (prepare, forward) collective blocks (e.g. moe.fused_experts_reduce): run
            # the miner's weight-prep on THIS rank's shard before the forward.
            prepared = None
            if prepare_name and slot.invoke_prepare is not None:
                prepared = slot.invoke_prepare(callable_from(module, prepare_name), inputs)

            # Validator-owned output buffer(s). Single-output slots keep the original
            # tensor-valued call shape; multi-output slots (e.g. ar_residual_rmsnorm's
            # [norm_out, new_residual]) receive the list.
            outs = [torch.empty(s, dtype=dtype, device=dev) for s in slot.out_shapes(inputs)]
            out_arg = outs[0] if len(outs) == 1 else outs

            invoke(entry, inputs, out_arg, dist.group.WORLD, prepared)  # miner fills the buffers
            staged.append((si, step, inputs, prepared, outs))

        # PHASE 2 — trusted references + comparison, after the whole burst: the fp32
        # cross-rank SUM of each rank's partial, then the slot's trusted post-reduce
        # math (collective_finish) if it does local work after the sum (residual add /
        # norm). No finish -> the sum IS the single expected output (bare all-reduce).
        passed, max_abs, score, detail, metric = True, 0.0, 1.0, "", "ratio"
        for si, step, inputs, prepared, outs in staged:
            partial = (slot.collective_partial(inputs, prepared)
                       if slot.collective_partial else inputs["x"].float())
            summed = partial.detach().float().clone()
            dist.all_reduce(summed, op=dist.ReduceOp.SUM)
            refs = (slot.collective_finish(inputs, summed, prepared)
                    if slot.collective_finish is not None else [summed])
            if len(refs) != len(outs):
                raise RuntimeError(
                    f"slot {slot.name}: collective_finish returned {len(refs)} reference(s) "
                    f"for {len(outs)} declared output(s)")

            for k, (out_k, ref_k) in enumerate(zip(outs, refs)):
                p_k, abs_k, _rel_k, s_k, d_k, metric = _compare(
                    out_k, ref_k, atol=tol.atol, rtol=tol.rtol, correctness=slot.correctness
                )
                passed = passed and bool(p_k)
                max_abs = max(max_abs, float(abs_k))
                if float(s_k) <= score:
                    prefix = f"step {si} {step}: " if len(steps) > 1 else ""
                    score = float(s_k)
                    detail = prefix + (f"out[{k}]: {d_k}" if len(outs) > 1 else d_k)
        verdict.update(passed=passed, score=score, max_abs=max_abs,
                       detail=detail, metric=metric)
    except BaseException:  # noqa: BLE001 - report any failure as a fail
        import traceback
        verdict["err"] = traceback.format_exc()
    finally:
        try:
            with open(os.path.join(result_dir, f"rank{rank}.pkl"), "wb") as f:
                pickle.dump(verdict, f)
        except Exception:  # noqa: BLE001
            pass
        try:
            import torch.distributed as dist
            if initialized and dist.is_initialized():
                dist.barrier()
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
    seed: int = 0,
    shapes: list[dict] | None = None,
    model_key: str | None = None,
    jitter_seed: int | None = None,
    bundle_path: str | None = None,
) -> VerifyResult:
    """Verify a collective slot's kernel across ``world_size`` spawned ranks.

    Defaults: ``device`` = cuda iff enough GPUs, else cpu; ``backend`` = nccl on cuda,
    gloo on cpu. gloo has no bf16, so the CPU path is forced to fp32. ``model_key`` selects
    the validator per-model slot profile (activation reference + metric); None -> generic.
    ``jitter_seed`` perturbs the count dims per run (same anti-shape-branching guard as
    the per-op verify — without it a collective kernel could hard-code the fixed verify
    shapes); jittered in the parent so every rank builds identical shapes.
    """
    import torch
    import torch.multiprocessing as mp

    from optima.verify import _jitter_shapes

    if device is None:
        device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() >= world_size) else "cpu"
    if backend is None:
        backend = "nccl" if device == "cuda" else "gloo"
    dtype_name = "float32" if backend == "gloo" else "bfloat16"  # gloo: no bf16
    test_shapes = shapes if shapes is not None else list(slot.shapes)
    if jitter_seed is not None:
        test_shapes = _jitter_shapes(test_shapes, jitter_seed)

    def _spawn_and_collect(shape_or_seq, label: dict, run_seed: int) -> ShapeResult:
        with tempfile.TemporaryDirectory(prefix="optima_collective_") as rd:
            init_method = f"file://{os.path.join(rd, 'pg_store')}"
            args = (world_size, backend, init_method, slot.name, source_path, entry_name,
                    shape_or_seq, dtype_name, device, run_seed, rd, prepare_name, model_key,
                    bundle_path)
            spawn_err = None
            try:
                mp.spawn(_rank_worker, args=args, nprocs=world_size, join=True)
            except Exception as exc:  # noqa: BLE001 - a crashed rank; verdicts still on disk
                spawn_err = repr(exc)

            verdicts = []
            for r in range(world_size):
                p = os.path.join(rd, f"rank{r}.pkl")
                if os.path.exists(p):
                    with open(p, "rb") as f:
                        verdicts.append(pickle.load(f))

        if not verdicts:
            return ShapeResult(shape=label, dtype=dtype_name, passed=False,
                               max_abs_err=float("inf"), max_rel_err=float("inf"),
                               pass_ratio=0.0, detail=f"no rank verdicts (spawn: {spawn_err})")
        rank_errs = [v for v in verdicts if v["err"]]
        passed = (len(verdicts) == world_size) and all(v["passed"] for v in verdicts) and not rank_errs
        worst = min(verdicts, key=lambda v: v["score"])
        detail = ""
        if rank_errs:
            detail = f"rank {rank_errs[0]['rank']} raised: " + rank_errs[0]["err"].strip().splitlines()[-1]
        elif not passed:
            detail = f"worst rank {worst['rank']}: {worst['metric']}={worst['score']:.4f} {worst['detail']}"
        return ShapeResult(
            shape=label, dtype=dtype_name, passed=passed,
            max_abs_err=max(v["max_abs"] for v in verdicts),
            max_rel_err=0.0, pass_ratio=worst["score"], detail=detail, metric=worst["metric"],
        )

    results: list[ShapeResult] = []
    for i, shape in enumerate(test_shapes):
        results.append(_spawn_and_collect(shape, shape, seed + i))

    # TEMPORAL sequence gate: the per-shape spawns above are clean-room single calls —
    # structurally blind to cross-call protocol state (IPC workspaces, per-token
    # counters, rotating sentinel buffers). Replay the same shapes big -> small -> big
    # inside ONE process: a kernel whose clears/rotation only maintain invariants for
    # the current call's rows is exactly right on every fresh call and corrupts the
    # first time the token count GROWS between calls (the 2026-07-07 engine failure;
    # ascending order would never trip it, grow-back does).
    seq = _temporal_sequence(test_shapes)
    if seq is not None:
        base = seq[: len(seq) // 3]
        label = {"sequence": "burst x3: num_tokens "
                             + "->".join(str(s.get("num_tokens")) for s in base)}
        results.append(_spawn_and_collect(seq, label, seed + 10_007))

    return VerifyResult(slot=slot.name, dtype=dtype_name, passed=all(r.passed for r in results) and bool(results),
                        shape_results=results)


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
