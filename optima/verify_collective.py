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
                 shape, dtype_name, device, seed, result_dir, prepare_name=None):
    """One rank: init the group, run the miner collective into a validator-owned buffer,
    compare to the trusted fp32 cross-rank reduce. Writes its verdict to ``result_dir``.

    Slot-driven via the slot's ``collective_partial`` (the fp32 tensor whose cross-rank
    SUM is the reference) and ``invoke_collective`` (how to call the kernel with the group),
    so this handles a bare all-reduce AND a block that owns its trailing reduce
    (moe.fused_experts_reduce) without hard-coding either contract."""
    import torch
    import torch.distributed as dist

    verdict = {"rank": rank, "passed": False, "score": 0.0, "max_abs": float("inf"),
               "detail": "", "metric": "ratio", "err": None}
    initialized = False
    try:
        from optima.sandbox import load_entry
        from optima.slots import get_slot

        if device == "cuda":
            torch.cuda.set_device(rank)
        dist.init_process_group(backend=backend, init_method=init_method, rank=rank, world_size=world_size)
        initialized = True

        slot = get_slot(slot_name)
        dtype = getattr(torch, dtype_name)
        dev = f"cuda:{rank}" if device == "cuda" else "cpu"
        inputs = slot.make_inputs(dtype=dtype, device=dev, seed=seed, rank=rank, world_size=world_size, **shape)

        # (prepare, forward) collective blocks (e.g. moe.fused_experts_reduce): run the
        # miner's weight-prep once on THIS rank's shard before the timed forward.
        prepared = None
        if prepare_name and slot.invoke_prepare is not None:
            prepared = slot.invoke_prepare(load_entry(source_path, prepare_name), inputs)

        out_shape = slot.out_shapes(inputs)[0]
        out = torch.empty(out_shape, dtype=dtype, device=dev)  # validator-owned output buffer

        entry = load_entry(source_path, entry_name)
        invoke = slot.invoke_collective or (lambda e, i, o, g, p: e(i["x"], o, g))
        invoke(entry, inputs, out, dist.group.WORLD, prepared)  # miner fills `out` with sum-over-ranks

        # Trusted high-precision reference: the fp32 cross-rank SUM of each rank's partial.
        partial = slot.collective_partial(inputs, prepared) if slot.collective_partial else inputs["x"].float()
        ref = partial.detach().float().clone()
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)

        tol = slot.tolerance_for(dtype)
        passed, max_abs, _max_rel, score, detail, metric = _compare(
            out, ref, atol=tol.atol, rtol=tol.rtol, correctness=slot.correctness
        )
        verdict.update(passed=bool(passed), score=float(score), max_abs=float(max_abs),
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
) -> VerifyResult:
    """Verify a collective slot's kernel across ``world_size`` spawned ranks.

    Defaults: ``device`` = cuda iff enough GPUs, else cpu; ``backend`` = nccl on cuda,
    gloo on cpu. gloo has no bf16, so the CPU path is forced to fp32.
    """
    import torch
    import torch.multiprocessing as mp

    if device is None:
        device = "cuda" if (torch.cuda.is_available() and torch.cuda.device_count() >= world_size) else "cpu"
    if backend is None:
        backend = "nccl" if device == "cuda" else "gloo"
    dtype_name = "float32" if backend == "gloo" else "bfloat16"  # gloo: no bf16
    test_shapes = shapes if shapes is not None else list(slot.shapes)

    results: list[ShapeResult] = []
    for i, shape in enumerate(test_shapes):
        with tempfile.TemporaryDirectory(prefix="optima_collective_") as rd:
            init_method = f"file://{os.path.join(rd, 'pg_store')}"
            args = (world_size, backend, init_method, slot.name, source_path, entry_name,
                    shape, dtype_name, device, seed + i, rd, prepare_name)
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
            results.append(ShapeResult(shape=shape, dtype=dtype_name, passed=False,
                                       max_abs_err=float("inf"), max_rel_err=float("inf"),
                                       pass_ratio=0.0, detail=f"no rank verdicts (spawn: {spawn_err})"))
            continue
        rank_errs = [v for v in verdicts if v["err"]]
        passed = (len(verdicts) == world_size) and all(v["passed"] for v in verdicts) and not rank_errs
        worst = min(verdicts, key=lambda v: v["score"])
        detail = ""
        if rank_errs:
            detail = f"rank {rank_errs[0]['rank']} raised: " + rank_errs[0]["err"].strip().splitlines()[-1]
        elif not passed:
            detail = f"worst rank {worst['rank']}: {worst['metric']}={worst['score']:.4f} {worst['detail']}"
        results.append(ShapeResult(
            shape=shape, dtype=dtype_name, passed=passed,
            max_abs_err=max(v["max_abs"] for v in verdicts),
            max_rel_err=0.0, pass_ratio=worst["score"], detail=detail, metric=worst["metric"],
        ))

    return VerifyResult(slot=slot.name, dtype=dtype_name, passed=all(r.passed for r in results) and bool(results),
                        shape_results=results)
