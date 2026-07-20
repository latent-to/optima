"""EXPERIMENTAL resident-engine kernel hot-swap hook (screen-tier probe).

BEFORE hook on ``ModelRunner.init_decode_cuda_graph``: when the validator sets
``OPTIMA_RESIDENT_SWAP`` to a control directory, each scheduler rank applies any
pending swap command in-process immediately before CUDA graphs are (re)captured.
The capture backend's eager warmups then JIT-compile the swapped kernel outside
the recording, and the recorded graphs bake the new kernel in.

This exists for the validator-owned resident SCREENING engine only: a persistent
model that evaluates a queue of candidate kernels without ever reloading weights
(swap -> recapture -> timed read). It is inert unless ``OPTIMA_RESIDENT_SWAP``
is set, which production qualification/crown paths never set.

Swap protocol (validator/host side):
  1. stage the bundle tree somewhere rank-readable;
  2. write ``$OPTIMA_RESIDENT_SWAP/command.json``:
       {"generation": <int, strictly increasing>, "bundle": "/abs/path" | null}
     (``null`` returns the engine to stock dispatch);
  3. trigger recapture on the live engine — sglang's own
     ``POST /update_weights_from_disk {"recapture_cuda_graph": true}`` path calls
     ``init_decode_cuda_graph()`` on every rank (model_runner.py), firing this hook.
Each rank writes ``$OPTIMA_RESIDENT_SWAP/ack.rank<R>.json`` recording the applied
generation, registered slots, and swap wall time; failures record the error and
leave the registry EMPTY+DISABLED (stock dispatch), never a half-swapped state.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import sys
import time

from optima.registry import REGISTRY, KernelRegistry

logger = logging.getLogger(__name__)

_MODULE = "sglang.srt.model_executor.model_runner"
_CLASS = "ModelRunner"
_METHOD = "init_decode_cuda_graph"
_HOOK_FLAG = "_optima_resident_swap"

# Trigger surface: sglang's /flush_cache request is broadcast to every TP rank's
# scheduler and is idle-gated, so a post-flush hook is a weight-free, all-rank
# recapture trigger (measured 2026-07-20: the update-weights trigger crashes the
# quantized M3 arena — minimax_m3_vl load_weights is not re-entrant, so the swap
# trigger must never touch weights).
_SCHED_MODULE = "sglang.srt.managers.scheduler"
_SCHED_CLASS = "Scheduler"
_SCHED_METHOD = "flush_cache"
_SCHED_HOOK_FLAG = "_optima_resident_swap_flush"

# Last generation this rank applied (process-global; one scheduler rank per process).
_applied_generation = -1


def _read_command(control_dir: str) -> tuple[int, str | None] | None:
    path = os.path.join(control_dir, "command.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return None
    except Exception:  # noqa: BLE001 - malformed command must not kill the engine
        logger.exception("optima: resident swap command unreadable at %s", path)
        return None
    generation = raw.get("generation") if isinstance(raw, dict) else None
    bundle = raw.get("bundle") if isinstance(raw, dict) else None
    if not isinstance(generation, int) or isinstance(generation, bool):
        return None
    if bundle is not None and (not isinstance(bundle, str) or not bundle.strip()):
        return None
    return generation, bundle


def _write_ack(control_dir: str, rank: object, payload: dict[str, object]) -> None:
    path = os.path.join(control_dir, f"ack.rank{rank}.json")
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, sort_keys=True)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - ack loss is diagnosable, not fatal
        logger.exception("optima: resident swap ack write failed at %s", path)


def _apply_pending_swap(model_runner: object, control_dir: str) -> None:
    global _applied_generation
    command = _read_command(control_dir)
    if command is None:
        return
    generation, bundle = command
    if generation <= _applied_generation:
        return
    rank = getattr(model_runner, "tp_rank", "unknown")
    started = time.perf_counter()
    ack: dict[str, object] = {
        "generation": generation,
        "bundle": bundle or "",
        "pid": os.getpid(),
    }
    try:
        from optima import seam

        result = seam.swap_resident_bundle(bundle)
        ack.update(result)
        ack["ok"] = True
    except Exception as exc:  # noqa: BLE001 - a bad bundle must not wedge the engine
        logger.exception("optima: resident swap failed for %s", bundle)
        REGISTRY.clear()
        ack["ok"] = False
        ack["error"] = str(exc)[:2048]
    _applied_generation = generation
    ack["swap_seconds"] = time.perf_counter() - started
    _write_ack(control_dir, rank, ack)


def _swap_pending(control_dir: str) -> bool:
    command = _read_command(control_dir)
    return command is not None and command[0] > _applied_generation


def install(registry: KernelRegistry = REGISTRY) -> None:
    """Install both hooks; inert unless OPTIMA_RESIDENT_SWAP is set.

    Hook 1 (swap): BEFORE ModelRunner.init_decode_cuda_graph — applies the
    pending bundle swap so the recapture bakes the new kernel in.
    Hook 2 (trigger): AFTER Scheduler.flush_cache — when the idle-gated flush
    succeeded and a swap command is pending, invoke init_decode_cuda_graph on
    this rank's live ModelRunner, which fires hook 1 then recaptures. The
    host triggers a swap by staging command.json then POSTing /flush_cache.
    """

    del registry
    control_dir = os.environ.get("OPTIMA_RESIDENT_SWAP", "").strip()
    if not control_dir:
        return

    mod = sys.modules.get(_MODULE)
    cls = getattr(mod, _CLASS, None) if mod is not None else None
    fn = getattr(cls, _METHOD, None) if cls is not None else None
    if fn is not None and not getattr(fn, _HOOK_FLAG, False):

        @functools.wraps(fn)
        def init_decode_cuda_graph(self, *args, **kwargs):
            if not getattr(self, "is_draft_worker", False):
                _apply_pending_swap(self, control_dir)
            return fn(self, *args, **kwargs)

        setattr(init_decode_cuda_graph, _HOOK_FLAG, True)
        init_decode_cuda_graph._optima_orig = fn  # type: ignore[attr-defined]
        setattr(cls, _METHOD, init_decode_cuda_graph)

    sched_mod = sys.modules.get(_SCHED_MODULE)
    sched_cls = getattr(sched_mod, _SCHED_CLASS, None) if sched_mod is not None else None
    sched_fn = getattr(sched_cls, _SCHED_METHOD, None) if sched_cls is not None else None
    if sched_fn is not None and not getattr(sched_fn, _SCHED_HOOK_FLAG, False):

        @functools.wraps(sched_fn)
        def flush_cache(self, *args, **kwargs):
            result = sched_fn(self, *args, **kwargs)
            if result and _swap_pending(control_dir):
                runner = getattr(
                    getattr(self, "tp_worker", None), "model_runner", None
                )
                if runner is not None:
                    try:
                        runner.init_decode_cuda_graph()
                    except Exception:  # noqa: BLE001 - keep the scheduler alive
                        logger.exception(
                            "optima: resident swap recapture failed after flush"
                        )
            return result

        setattr(flush_cache, _SCHED_HOOK_FLAG, True)
        flush_cache._optima_orig = sched_fn  # type: ignore[attr-defined]
        setattr(sched_cls, _SCHED_METHOD, flush_cache)


def uninstall() -> None:
    mod = sys.modules.get(_MODULE)
    cls = getattr(mod, _CLASS, None) if mod is not None else None
    fn = getattr(cls, _METHOD, None) if cls is not None else None
    if fn is None or not getattr(fn, _HOOK_FLAG, False):
        return
    setattr(cls, _METHOD, fn._optima_orig)


def is_installed() -> bool:
    mod = sys.modules.get(_MODULE)
    cls = getattr(mod, _CLASS, None) if mod is not None else None
    fn = getattr(cls, _METHOD, None) if cls is not None else None
    return bool(fn is not None and getattr(fn, _HOOK_FLAG, False))
