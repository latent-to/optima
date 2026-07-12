"""Shared engine-launch context manager used by the eval modules.

Centralizes the spawn-safe, tamper-resistant launch: mark this process as the
driver (so it never imports miner code), set the seam env, build the sglang
Engine, and clean it up. Both the KL eval and the benchmark eval use this.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any, Optional

logger = logging.getLogger("optima.eval")


class IsolationError(RuntimeError):
    """Raised when candidate isolation was requested but could not be proven."""


@contextmanager
def env(**overrides: str):
    saved = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def isolate_network() -> bool:
    """Put THIS process (and every child it spawns) into a fresh network namespace with
    NO egress, so untrusted miner code can't reach an external API to fake the output.
    Loopback is brought up so sglang's localhost IPC still works; the model is forced
    offline (it must already be cached). Self-checks that egress is actually gone.

    This is the boundary that makes the framework-mode token-match gate cheat-PROOF: the
    candidate must compute the right tokens — it can't see the trusted reference
    (separate process) and now can't fetch it either. Requires CAP_SYS_ADMIN (run the
    GPU box privileged; chain/cloud secrets live on a separate CPU control box). Returns
    True iff the candidate is confirmed no-egress; logs loudly and returns False if not.
    """
    import subprocess

    clone_newnet = getattr(os, "CLONE_NEWNET", None)
    if clone_newnet is None or not hasattr(os, "unshare"):
        logger.warning("optima: os.unshare/CLONE_NEWNET unavailable (need py>=3.12); candidate NOT isolated")
        return False
    try:
        os.unshare(clone_newnet)  # fresh netns: only `lo`, which starts DOWN
    except OSError as exc:
        logger.warning("optima: network isolation failed (%s); candidate NOT no-egress", exc)
        return False
    # Bring up loopback (the sglang scheduler<->detokenizer IPC uses localhost); external
    # stays unreachable because the netns has no route off-box.
    try:
        subprocess.run(["ip", "link", "set", "lo", "up"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("optima: could not bring up netns loopback (%s); sglang IPC may fail", exc)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Self-check: prove egress is actually gone (a fail-closed signal in the log).
    import socket

    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
        logger.error("optima: ISOLATION FAILED — candidate still has network egress!")
        return False
    except OSError:
        logger.warning("optima: candidate network-isolated (no egress; loopback only)")
        return True


def _offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


def prepare_candidate_environment(cfg, *, bundle_path: str, active: bool) -> None:
    """Apply candidate-only process isolation/rebuild work before importing SGLang."""
    if not active:
        return
    framework_mode = getattr(cfg, "framework_mode", False)
    isolate = getattr(cfg, "isolate", False)
    allow_unsafe = getattr(cfg, "allow_unsafe_no_isolation", False)
    has_setup = False
    if bundle_path:
        from optima.manifest import load_manifest

        manifest = load_manifest(bundle_path)
        has_setup = any(op.setup for op in manifest.ops)
        if has_setup and not framework_mode:
            raise IsolationError(
                "bundle declares setup() but framework_mode is not enabled. "
                "Engine-wide mutation requires external token fidelity and isolation."
            )
    if framework_mode and not isolate:
        if has_setup:
            raise IsolationError(
                "setup() requires proven no-egress candidate isolation; "
                "the unsafe development override cannot arm engine-wide mutation"
            )
        if not allow_unsafe:
            raise IsolationError(
                "framework_mode requires no-egress candidate isolation. "
                "Use --allow-unsafe-no-isolation only for local throughput debugging."
            )
        logger.error(
            "optima: UNSAFE dev override: framework-mode candidate is running without "
            "requested network isolation"
        )
        _offline_env()
    if isolate:
        if not isolate_network():
            if has_setup:
                raise IsolationError(
                    "setup() requires proven no-egress candidate isolation; "
                    "the unsafe development override cannot bypass a failed fence"
                )
            if not allow_unsafe:
                raise IsolationError(
                    "candidate network isolation was requested but could not be proven. "
                    "Run the eval worker with CAP_SYS_ADMIN/CAP_NET_ADMIN, or inside a "
                    "container/VM whose candidate process has no network egress. "
                    "Use --allow-unsafe-no-isolation only for local throughput debugging."
                )
            logger.error(
                "optima: UNSAFE dev override: candidate network isolation failed; "
                "continuing with egress possible"
            )
            _offline_env()
    if bundle_path:
        from optima.rebuild import apply_rebuild_plan

        if apply_rebuild_plan(bundle_path):
            logger.warning("optima: applied rebuild plan for %s", bundle_path)
        _dep_overlay_env(bundle_path)


def _dep_overlay_env(bundle_path: str) -> None:
    """Candidate-local JIT workspace for a dep-patched candidate.

    ``FLASHINFER_WORKSPACE_BASE`` is read ONCE at ``flashinfer.jit.env`` import (a real
    ``os.getenv``, unlike everything else there — verified 2026-07-07), so it must be a
    process env var set BEFORE the engine spawns; the overlay integration cannot rebind
    it later. Without this, a patched JIT build and a stock JIT build of the same
    module name share a cache dir — ninja does invalidate on the changed source path,
    but concurrent candidates would serialize/race on the shared build files.
    """
    import os

    from optima.manifest import load_manifest

    manifest = load_manifest(bundle_path)
    if manifest.dep_patches and "FLASHINFER_WORKSPACE_BASE" not in os.environ:
        raise RuntimeError(
            "dep-patched candidate lacks its content-addressed FlashInfer workspace; "
            "the reviewed rebuild phase did not provision the runtime environment"
        )


def engine_kwargs(cfg, *, active: bool = False) -> dict[str, Any]:
    """Translate an ``EvalConfig`` into ``sglang.Engine`` kwargs.

    Shared by both eval paths so multi-GPU knobs (``tp_size`` / ``moe_runner_backend``
    / ``disable_custom_all_reduce``) and deterministic mode apply identically. New
    fields are read with ``getattr`` so an older/duck-typed cfg still works.
    """
    kwargs: dict[str, Any] = dict(
        model_path=cfg.model_path,
        dtype=cfg.dtype,
        mem_fraction_static=cfg.mem_fraction_static,
        random_seed=cfg.seed,
        log_level=cfg.log_level,
    )
    # Only pass these when explicitly set so sglang keeps its strong production
    # defaults otherwise (auto attention backend + CUDA graphs ON). A weak baseline
    # lets miners win against a crippled reference.
    attention_backend = getattr(cfg, "attention_backend", None)
    if active and getattr(cfg, "candidate_attention_backend", None):
        attention_backend = cfg.candidate_attention_backend
    if attention_backend:
        kwargs["attention_backend"] = attention_backend
    if getattr(cfg, "disable_cuda_graph", False):
        kwargs["disable_cuda_graph"] = True
    if getattr(cfg, "deterministic", False):
        kwargs["enable_deterministic_inference"] = True
    if getattr(cfg, "tp_size", None):
        kwargs["tp_size"] = int(cfg.tp_size)
    if getattr(cfg, "max_running_requests", None):
        kwargs["max_running_requests"] = int(cfg.max_running_requests)
    moe_runner_backend = getattr(cfg, "moe_runner_backend", None)
    if active and getattr(cfg, "candidate_moe_runner_backend", None):
        moe_runner_backend = cfg.candidate_moe_runner_backend
    if moe_runner_backend:
        kwargs["moe_runner_backend"] = moe_runner_backend
    disable_custom_all_reduce = getattr(cfg, "disable_custom_all_reduce", False)
    if active and getattr(cfg, "candidate_disable_custom_all_reduce", None) is not None:
        disable_custom_all_reduce = cfg.candidate_disable_custom_all_reduce
    if disable_custom_all_reduce:
        kwargs["disable_custom_all_reduce"] = True
    kwargs.update(getattr(cfg, "extra_engine_kwargs", {}) or {})
    if active:
        kwargs.update(getattr(cfg, "candidate_extra_engine_kwargs", {}) or {})
    return kwargs


def _sweep_gpu_procs() -> int:
    """Kill every OTHER process in this namespace holding an nvidia device fd.

    Failed engine launches can strand scheduler subprocesses that survive both
    the launch child's reap and sglang's own kill cascade (they re-session), each
    pinning the model's full VRAM. Only ever called from a launch subprocess that
    has not created its engine yet, and only when OPTIMA_GPU_SWEEP=1 — a dedicated
    eval box where everything on the visible GPUs belongs to this evaluation.
    """
    import signal

    me = os.getpid()
    killed = 0
    for pid_dir in os.listdir("/proc"):
        if not pid_dir.isdigit() or int(pid_dir) == me:
            continue
        fd_dir = f"/proc/{pid_dir}/fd"
        try:
            for fd in os.listdir(fd_dir):
                try:
                    if os.readlink(f"{fd_dir}/{fd}").startswith("/dev/nvidia"):
                        os.kill(int(pid_dir), signal.SIGKILL)
                        killed += 1
                        break
                except OSError:
                    continue
        except OSError:  # process exited, or not ours to inspect
            continue
    if killed:
        logger.warning("optima: GPU sweep killed %d stranded process(es)", killed)
    return killed


def _wait_gpu_drain(threshold_mib: int = 4096, timeout_s: float = 150.0) -> None:
    """Block until every visible GPU is under ``threshold_mib`` used, or timeout.

    Evaluate runs engine launches back-to-back out of subprocesses; the previous
    launch's schedulers release their VRAM a beat after the driver regains control
    (and a wedged shutdown can pin the whole model until the reap in the launch
    finally fires). Sizing the next KV pool against that residue OOMs at startup.
    Polls nvidia-smi (never initializes CUDA in this process); on timeout, warns
    and proceeds — the guard must never fail a run on its own.
    """
    import subprocess
    import time

    deadline = time.monotonic() + timeout_s
    sweep_at = time.monotonic() + 25.0  # give a clean shutdown a fair head start
    swept = os.environ.get("OPTIMA_GPU_SWEEP") != "1"  # disabled -> pretend done
    # Scope the wait to THIS launch's GPUs: on a shared box another lane's engine
    # legitimately holds its own devices for the whole run — without the filter
    # every launch here would stall out the full timeout staring at it.
    query = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        query += ["-i", cvd]
    last = ""
    while time.monotonic() < deadline:
        try:
            out = subprocess.run(query, capture_output=True, text=True, timeout=10).stdout
            used = [int(x) for x in out.split()]
        except Exception:  # noqa: BLE001 — no/odd nvidia-smi: nothing to wait for
            return
        if not used or max(used) < threshold_mib:
            return
        if not swept and time.monotonic() >= sweep_at:
            _sweep_gpu_procs()
            swept = True
        last = ",".join(map(str, used))
        time.sleep(2.0)
    logger.warning("optima: GPUs did not drain below %d MiB within %.0fs (used MiB: %s); "
                   "launching anyway", threshold_mib, timeout_s, last)


def _active_execution_members(
    active_receipts: list[dict], *, expected_member_count: int
) -> list[str]:
    """Validate active scheduler membership and return the identical slot set."""
    pids: list[int] = []
    slot_sets: list[tuple[str, ...]] = []
    for receipt in active_receipts:
        pid = receipt.get("pid")
        slots = receipt.get("slots")
        if type(pid) is not int or pid < 1:
            raise RuntimeError("candidate engine launch: malformed active-member PID")
        if (
            not isinstance(slots, list)
            or not slots
            or any(not isinstance(slot, str) or not slot for slot in slots)
            or len(set(slots)) != len(slots)
        ):
            raise RuntimeError("candidate engine launch: malformed active slot set")
        pids.append(pid)
        slot_sets.append(tuple(sorted(slots)))
    if len(set(pids)) != len(pids):
        raise RuntimeError("candidate engine launch: duplicate active-member PID")
    if len(pids) != expected_member_count:
        raise RuntimeError(
            "candidate engine launch: incomplete active-member coverage "
            f"({len(pids)}/{expected_member_count}); refusing a partially active engine"
        )
    if not slot_sets or len(set(slot_sets)) != 1:
        raise RuntimeError(
            "candidate engine launch: scheduler members disagree on registered slots"
        )
    return list(slot_sets[0])


def _require_execution_completion(
    receipt_dir: str,
    *,
    active_receipts: list[dict],
    expected_slots: list[str],
    expected_member_count: int,
) -> str:
    """Fail closed unless every active member completed every registered slot."""
    from optima import receipts

    completed = receipts.collect(receipt_dir, "completed")
    fallbacks = receipts.collect(receipt_dir, "fallback")
    passed, detail = receipts.completed_gate(
        completed,
        expected_slots=expected_slots,
        member_receipts=active_receipts,
        expected_member_count=expected_member_count,
        fallback_receipts=fallbacks,
    )
    if not passed:
        raise RuntimeError(
            "candidate engine run failed execution coverage: " + detail
        )
    return detail


@contextmanager
def launched_engine(cfg, *, bundle_path: str, active: bool,
                    audit_rate: float = 0.0, audit_out: Optional[list] = None):
    """Launch a sglang Engine with the Optima seam configured.

    ``cfg`` is an ``EvalConfig`` (see optima.eval.throughput_kl). The miner
    kernel runs only in the spawned scheduler child; THIS process is marked as
    the driver so it never imports miner code (timing stays tamper-resistant).

    An ACTIVE launch demands seam receipts (see optima/receipts.py): at least one
    scheduler rank must report the bundle loaded+enabled before we hand the engine
    to the caller. After generation, every active scheduler member must report a
    completed model-facing output for every registered slot, with zero selected-path
    fallbacks. ``fired`` remains routing-only: lookup may precede a later stock route
    or exception. These receipts prevent accidental phantom execution but are still
    forgeable process-local diagnostics until complete-engine isolation lands.

    ``audit_rate > 0`` arms the IN-ENGINE AUDIT (optima/audit.py) in the ranks:
    sampled dispatcher calls are re-run through the captured stock baseline and
    compared under the slot's verify tolerances. Only ever set on an UNTIMED
    quality launch — audited calls carry clone+baseline overhead. The rolling
    per-rank audit receipts are appended to ``audit_out`` before cleanup. The
    sampling seed is fixed per launch and shared by all ranks (collective
    baselines need rank-identical sampling; see audit.py).
    """
    import random
    import shutil
    import tempfile

    from optima import receipts, seam

    seam.mark_driver()
    prepare_candidate_environment(cfg, bundle_path=bundle_path, active=active)
    receipt_dir = tempfile.mkdtemp(prefix="optima_receipts_") if active else ""
    # Explicitly clear an ambient directory for baseline launches so they cannot
    # contaminate a candidate's fresh evidence namespace.
    extra_env = {"OPTIMA_SEAM_RECEIPT_DIR": receipt_dir if active else ""}
    if active and audit_rate > 0.0:
        extra_env["OPTIMA_SLOT_AUDIT"] = f"{audit_rate:g}"
        extra_env["OPTIMA_SLOT_AUDIT_SEED"] = str(random.SystemRandom().randrange(2**31))
    try:
        with env(
            OPTIMA_BUNDLE_PATH=bundle_path or "",
            OPTIMA_ACTIVE="1" if active else "0",
            # Only the trusted active-launch decision may arm setup(). Force an
            # explicit zero for ordinary candidates and every baseline so an
            # ambient parent variable cannot widen the submission lane.
            OPTIMA_FRAMEWORK_MODE=(
                "1" if active and getattr(cfg, "framework_mode", False) else "0"
            ),
            SGLANG_PLUGINS="optima",
            **extra_env,
        ):
            import sglang as sgl

            _wait_gpu_drain()
            resolved_engine_kwargs = engine_kwargs(cfg, active=active)
            engine = sgl.Engine(**resolved_engine_kwargs)
            ok = False
            active_receipts: list[dict] = []
            expected_slots: list[str] = []
            expected_member_count = int(
                resolved_engine_kwargs.get("tp_size", 1) or 1
            )
            try:
                if active:
                    active_receipts = receipts.require(
                        receipt_dir, "active", context="candidate engine launch"
                    )
                    expected_slots = _active_execution_members(
                        active_receipts,
                        expected_member_count=expected_member_count,
                    )
                    logger.info("optima: seam active receipts: %s", active_receipts)
                yield engine
                ok = True
            finally:
                try:
                    engine.shutdown()
                except Exception:  # noqa: BLE001
                    pass
                # Some builds' Engine.shutdown() leaves scheduler subprocesses
                # alive, each pinning the model's whole VRAM — which starves the
                # NEXT launch's pool sizing (measured 2026-07-10: B' startup OOM
                # behind 4x180GB orphaned schedulers). This launch subprocess owns
                # every engine process, so reap the remaining tree before handing
                # control back to the driver.
                try:
                    from sglang.srt.utils import kill_process_tree

                    kill_process_tree(os.getpid(), include_parent=False)
                except Exception:  # noqa: BLE001
                    pass
                if active and ok and audit_out is not None:
                    audit_out.extend(receipts.collect(receipt_dir, "audit"))
                if active and ok:
                    detail = _require_execution_completion(
                        receipt_dir,
                        active_receipts=active_receipts,
                        expected_slots=expected_slots,
                        expected_member_count=expected_member_count,
                    )
                    logger.info("optima: %s", detail)
    finally:
        if receipt_dir:
            shutil.rmtree(receipt_dir, ignore_errors=True)


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
