"""Shared engine-worker policy helpers.

This module is safe for both the legacy development launcher and the isolated
OCI worker to import.  It contains no engine construction, subprocess launch,
or grading authority.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


logger = logging.getLogger("optima.eval.engine-worker")


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _loopback_is_up() -> bool:
    import fcntl
    import socket
    import struct

    request = struct.Struct("16sH14s")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            response = fcntl.ioctl(
                sock.fileno(), 0x8913, request.pack(b"lo", 0, b"")
            )
        _name, flags, _padding = request.unpack(response)
        return bool(flags & 0x1)
    except (OSError, ValueError, struct.error):
        return False


def _network_namespace_is_loopback_only() -> bool:
    """Return whether the current Linux network namespace exposes only loopback."""

    try:
        interfaces = {
            line.split(":", 1)[0].strip()
            for line in Path("/proc/net/dev").read_text().splitlines()[2:]
            if ":" in line
        }
        if interfaces != {"lo"}:
            return False
        ipv4 = Path("/proc/net/route").read_text().splitlines()[1:]
        if any(line.split()[0] != "lo" for line in ipv4 if line.split()):
            return False
        ipv6 = Path("/proc/net/ipv6_route").read_text().splitlines()
        if any(line.split()[-1] != "lo" for line in ipv6 if line.split()):
            return False
    except (OSError, IndexError):
        return False
    return True


def _egress_is_blocked() -> bool:
    import socket

    try:
        socket.create_connection(("1.1.1.1", 443), timeout=2).close()
        return False
    except OSError:
        return True


def _process_sandbox_is_hardened() -> bool:
    """Verify that the live worker cannot elevate or mutate its root filesystem."""

    try:
        status: dict[str, str] = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            key, separator, value = line.partition(":")
            if separator:
                status[key] = value.strip()
        effective = int(status["CapEff"], 16)
        bounding = int(status["CapBnd"], 16)
        no_new_privileges = int(status["NoNewPrivs"])
        seccomp_mode = int(status["Seccomp"])
        seccomp_filters = int(status["Seccomp_filters"])
        ptrace_scope = int(Path("/proc/sys/kernel/yama/ptrace_scope").read_text())
        root_options = None
        for line in Path("/proc/mounts").read_text().splitlines():
            fields = line.split()
            if len(fields) >= 4 and fields[1] == "/":
                root_options = set(fields[3].split(","))
                break
    except (OSError, KeyError, ValueError):
        return False
    return (
        effective == 0
        and bounding == 0
        and no_new_privileges == 1
        and seccomp_mode == 2
        and seccomp_filters >= 1
        and ptrace_scope >= 1
        and root_options is not None
        and "ro" in root_options
    )


def _path_mount_is_read_only(path: str) -> bool:
    try:
        return bool(os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1))
    except OSError:
        return False


def engine_kwargs(cfg, *, active: bool = False) -> dict[str, Any]:
    """Translate a development ``EvalConfig`` into ``sglang.Engine`` kwargs."""

    kwargs: dict[str, Any] = dict(
        model_path=cfg.model_path,
        dtype=cfg.dtype,
        mem_fraction_static=cfg.mem_fraction_static,
        random_seed=cfg.seed,
        log_level=cfg.log_level,
    )
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
    aot_loaded = receipts.collect(receipt_dir, "aot_loaded")
    aot_invoked = receipts.collect(receipt_dir, "aot_invoked")
    passed, detail = receipts.completed_gate(
        completed,
        expected_slots=expected_slots,
        member_receipts=active_receipts,
        expected_member_count=expected_member_count,
        fallback_receipts=fallbacks,
    )
    if not passed:
        observed = (
            f"observed_receipts=completed:{len(completed)},fallback:{len(fallbacks)},"
            f"aot_loaded:{len(aot_loaded)},aot_invoked:{len(aot_invoked)}"
        )
        raise RuntimeError(
            "candidate engine run failed execution coverage: "
            + detail
            + "; "
            + observed
        )
    if aot_invoked and not aot_loaded:
        raise RuntimeError(
            "candidate engine run has sealed CuTe AOT use evidence without "
            "matching load evidence"
        )
    if aot_loaded:
        aot_slots = sorted(
            {
                row.get("slot")
                for row in aot_loaded
                if isinstance(row.get("slot"), str) and row.get("slot")
            }
        )
        if not aot_slots:
            raise RuntimeError(
                "candidate engine run has malformed CuTe AOT load evidence"
            )
        if not set(aot_slots).issubset(expected_slots):
            raise RuntimeError(
                "candidate engine run loaded sealed CuTe AOT for an inactive slot"
            )
        loaded_passed, loaded_detail = receipts.completed_gate(
            aot_loaded,
            expected_slots=aot_slots,
            member_receipts=active_receipts,
            expected_member_count=expected_member_count,
        )
        if not loaded_passed:
            raise RuntimeError(
                "candidate engine run failed sealed CuTe AOT load coverage: "
                + loaded_detail
            )
        aot_passed, aot_detail = receipts.completed_gate(
            aot_invoked,
            expected_slots=aot_slots,
            member_receipts=active_receipts,
            expected_member_count=expected_member_count,
        )
        if not aot_passed:
            raise RuntimeError(
                "candidate engine run failed sealed CuTe AOT use coverage: "
                + aot_detail
            )
        detail += "; sealed CuTe AOT " + loaded_detail + "; " + aot_detail
    return detail


@dataclass(frozen=True)
class EngineWorkerHandle:
    engine: object
    require_completion: Any


@contextlib.contextmanager
def _environment(**overrides: str) -> Iterator[None]:
    saved = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def isolated_engine_session(
    cfg: object,
    *,
    bundle_path: str,
    active: bool,
    framework_mode: bool,
    install_seams: bool = True,
) -> Iterator[EngineWorkerHandle]:
    """Construct one engine only inside the already-proven OCI worker fence."""

    if (
        not _truthy_env("OPTIMA_EXTERNAL_NO_EGRESS")
        or not _truthy_env("OPTIMA_ENGINE_WORKER")
        or not all((
            _loopback_is_up(),
            _network_namespace_is_loopback_only(),
            _egress_is_blocked(),
            _process_sandbox_is_hardened(),
        ))
        or (framework_mode and not active)
        or (active and not install_seams)
    ):
        raise RuntimeError("isolated engine session lacks its trusted OCI fence")
    receipts = None
    if install_seams:
        from optima import receipts as receipt_module, seam

        seam.mark_driver()
        receipts = receipt_module
    from optima.seams import seam_binding_environment

    gate_environment = seam_binding_environment(
        getattr(cfg, "seam_bindings", ()) if install_seams else ()
    )
    receipt_dir = tempfile.mkdtemp(prefix="optima_receipts_") if active else ""
    try:
        session_environment = {
            "OPTIMA_ACTIVE": "1" if active else "0",
            "OPTIMA_BUNDLE_PATH": bundle_path if active else "",
            "OPTIMA_FRAMEWORK_MODE": "1" if framework_mode else "0",
            "OPTIMA_SEAM_RECEIPT_DIR": receipt_dir,
            "OPTIMA_SLOT_AUDIT": "",
            "OPTIMA_SLOT_AUDIT_SEED": "",
            "SGLANG_PLUGINS": "optima" if install_seams else "",
            **gate_environment,
        }
        with _environment(**session_environment):
            import sglang as sgl

            kwargs = engine_kwargs(cfg, active=active)
            engine = sgl.Engine(**kwargs)
            active_receipts: list[dict] = []
            expected_slots: list[str] = []
            expected_members = int(kwargs.get("tp_size", 1) or 1)
            try:
                if active:
                    assert receipts is not None
                    active_receipts = receipts.require(
                        receipt_dir, "active", context="candidate engine launch"
                    )
                    expected_slots = _active_execution_members(
                        active_receipts, expected_member_count=expected_members
                    )

                def complete() -> None:
                    if active:
                        _require_execution_completion(
                            receipt_dir,
                            active_receipts=active_receipts,
                            expected_slots=expected_slots,
                            expected_member_count=expected_members,
                        )

                yield EngineWorkerHandle(engine, complete)
            finally:
                try:
                    engine.shutdown()
                except Exception:  # noqa: BLE001 - force-reap follows
                    pass
                try:
                    from sglang.srt.utils import kill_process_tree

                    kill_process_tree(os.getpid(), include_parent=False)
                except Exception:  # noqa: BLE001 - outer OCI teardown remains authoritative
                    pass
    finally:
        if receipt_dir:
            shutil.rmtree(receipt_dir, ignore_errors=True)
