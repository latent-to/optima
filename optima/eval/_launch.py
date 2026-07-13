"""Spawn-safe subprocess call helper for the CLI verify path.

Each model/kernel load must run in its own process: sglang + deterministic mode
set process-global state (torch deterministic algorithms, the cuBLAS workspace,
the sampling backend) and hold a CUDA context, and the trusted CLI process must
never import miner code. ``cmd_verify`` runs each candidate load through
``call_in_subprocess``.

The engine-launch machinery that used to live here (isolation fences,
``launched_engine``) belongs to the isolated OCI execution path now —
``optima/eval/engine_worker.py`` (in-worker session + isolation probes) and
``optima/eval/oci_session_worker.py`` (the no-egress worker).
"""

from __future__ import annotations


def _subprocess_entry(out_path, fn, args, kwargs):
    """Run ``fn(*args, **kwargs)`` and pickle the result (or traceback) to a file."""
    import pickle
    import traceback

    try:
        payload = {"value": fn(*args, **kwargs), "error": None}
    except BaseException:  # noqa: BLE001 - report ANY failure back to the parent
        payload = {"value": None, "error": traceback.format_exc()}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)


def call_in_subprocess(fn, *args, **kwargs):
    """Run ``fn(*args, **kwargs)`` in a FRESH spawned process; return its result.

    Each model launch must run in its own process. sglang + deterministic mode set
    process-global state (torch deterministic algorithms, the cuBLAS workspace, the
    sampling backend) and hold a CUDA context; a second launch in the same driver
    process inherits that state and — observed on gpt-oss-120b in deterministic mode —
    the candidate launch then produces NaN/garbage. A fresh process makes the baseline
    and candidate launches independent and frees all GPU/host memory between them.

    ``fn`` must be a module-level (picklable) callable; the result travels back through
    a temp pickle file (avoids mp.Queue size limits / pipe deadlocks on large logprob
    payloads). Raises ``RuntimeError`` if the child crashes or ``fn`` raises.
    """
    import multiprocessing as mp
    import os
    import pickle
    import tempfile

    ctx = mp.get_context("spawn")
    fd, path = tempfile.mkstemp(prefix="optima_launch_", suffix=".pkl")
    os.close(fd)
    try:
        proc = ctx.Process(target=_subprocess_entry, args=(path, fn, args, kwargs))
        proc.start()
        proc.join()
        try:
            with open(path, "rb") as f:
                payload = pickle.load(f)
        except (EOFError, FileNotFoundError, pickle.UnpicklingError) as exc:
            raise RuntimeError(
                f"launch subprocess crashed (exitcode={proc.exitcode}) with no result: {exc}"
            ) from None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if payload.get("error"):
        raise RuntimeError("launch subprocess failed:\n" + payload["error"])
    return payload["value"]
