from __future__ import annotations

import hashlib
import json
import os
import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

import optima.eval.oci_process as process_mod
from optima.eval.oci_process import (
    ATTACHED_STDERR_MAX_BYTES,
    CommandResult,
    GPU_RESERVATION_ENV,
    GPU_RESERVATION_LABEL,
    OCIAttachedDiagnostic,
    OCIStderrArtifactReceipt,
    OCIQuiescenceReceipt,
    OCIProcessError,
    OCIProcessManager,
    OCIProcessTimeout,
)


CONTAINER_ID = "d" * 64


@pytest.fixture(autouse=True)
def _clear_gpu_reservation_env(monkeypatch) -> None:
    monkeypatch.delenv(GPU_RESERVATION_ENV, raising=False)


class Commands:
    def __init__(self) -> None:
        self.rows: list[tuple[str, ...]] = []
        self.present: set[str] = set()
        self.labels: dict[str, tuple[str, str]] = {}
        self.gpu_labels: dict[str, str] = {}

    def __call__(self, argv, *, timeout_s, max_output_bytes):
        row = tuple(argv)
        self.rows.append(row)
        if row[1:3] == ("container", "ls"):
            names = [value for value in row if value.startswith("name=^/")]
            present = bool(self.present) if not names else names[0][7:-1] in self.present
            return CommandResult(0, (CONTAINER_ID + "\n").encode() if present else b"", b"")
        if row[1:3] == ("container", "inspect"):
            name = next(iter(self.present))
            executor, lease = self.labels.get(name, ("validator-a", "lease-1"))
            labels = {
                "optima.executor_id": executor,
                "optima.lease_id": lease,
            }
            if name in self.gpu_labels:
                labels[GPU_RESERVATION_LABEL] = self.gpu_labels[name]
            payload = {
                "Id": CONTAINER_ID,
                "Name": f"/{name}",
                "Labels": labels,
            }
            return CommandResult(0, json.dumps(payload).encode(), b"")
        if row[1:3] == ("rm", "--force"):
            self.present.clear()
        return CommandResult(0, b"", b"")


class FakeProcess:
    next_pid = 4000

    def __init__(self, argv, **kwargs):
        self.argv = tuple(argv)
        self.kwargs = kwargs
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.returncode = None
        self.input = None
        self.waits = []
        self.terminated = False
        self.killed = False

    def communicate(self, *, input, timeout):
        self.input = input
        self.waits.append(timeout)
        self.returncode = 0

    def wait(self, timeout):
        self.waits.append(timeout)
        if self.returncode is None:
            self.returncode = -9 if self.killed else -15 if self.terminated else 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def poll(self):
        return self.returncode


class FakeStream:
    next_fd = 80

    def __init__(self) -> None:
        self.closed = False
        self.fd = FakeStream.next_fd
        FakeStream.next_fd += 1

    def fileno(self) -> int:
        return self.fd

    def close(self) -> None:
        self.closed = True


def _manager(tmp_path: Path, commands: Commands | None = None) -> OCIProcessManager:
    return OCIProcessManager(
        docker_binary="/usr/bin/docker",
        recovery_root=tmp_path / "recovery",
        executor_id="validator-a",
        runner=commands or Commands(),
    )


def test_register_writes_exact_lease_and_run_prefix(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(
        lease_id="lease-1",
        container_name="optima-prebuild-1",
        mount_relpaths=("mounts/work",),
        stage_relpaths=("stages/output",),
    )
    assert json.loads(lease.record_path.read_text()) == {
        "schema": "optima.oci-process-lease.v1",
        "executor_id": "validator-a",
        "lease_id": "lease-1",
        "container_name": "optima-prebuild-1",
        "mount_relpaths": ["mounts/work"],
        "stage_relpaths": ["stages/output"],
    }
    assert lease.run_prefix(manager.docker_binary) == (
        "/usr/bin/docker",
        "run",
        "--name=optima-prebuild-1",
        f"--cidfile={lease.cid_path}",
        "--label=optima.executor_id=validator-a",
        "--label=optima.lease_id=lease-1",
    )
    assert lease.record_path.stat().st_mode & 0o777 == 0o600


def test_register_propagates_gpu_reservation_label(
    tmp_path: Path, monkeypatch
) -> None:
    reservation_id = "a1b2c3d4e5f6"
    monkeypatch.setenv(GPU_RESERVATION_ENV, reservation_id)
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")

    assert lease.gpu_reservation_id == reservation_id
    assert json.loads(lease.record_path.read_text())["gpu_reservation_id"] == reservation_id
    assert lease.run_prefix(manager.docker_binary) == (
        "/usr/bin/docker",
        "run",
        "--name=container-1",
        f"--cidfile={lease.cid_path}",
        "--label=optima.executor_id=validator-a",
        "--label=optima.lease_id=lease-1",
        f"--label={GPU_RESERVATION_LABEL}={reservation_id}",
    )
    monkeypatch.delenv(GPU_RESERVATION_ENV)
    restored = manager._lease_from_record(lease.record_path)
    assert restored.gpu_reservation_id == reservation_id
    assert restored.run_prefix(manager.docker_binary) == lease.run_prefix(
        manager.docker_binary
    )


def test_register_rejects_malformed_gpu_reservation_identity(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(GPU_RESERVATION_ENV, "not-a-reservation")
    manager = _manager(tmp_path)
    with pytest.raises(OCIProcessError, match=GPU_RESERVATION_ENV):
        manager.register(lease_id="lease-1", container_name="container-1")


def test_cleanup_requires_matching_gpu_reservation_label(
    tmp_path: Path, monkeypatch
) -> None:
    reservation_id = "a1b2c3d4e5f6"
    monkeypatch.setenv(GPU_RESERVATION_ENV, reservation_id)
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    commands.present.add("container-1")
    commands.gpu_labels["container-1"] = "000000000000"

    with pytest.raises(OCIProcessError, match="exact lease labels"):
        manager.force_remove_container(lease)


def test_quiescence_receipt_requires_empty_executor_namespace(tmp_path: Path) -> None:
    commands = Commands()
    times = iter((10.0, 11.0))
    manager = OCIProcessManager(
        docker_binary="/usr/bin/docker",
        recovery_root=tmp_path / "recovery",
        executor_id="validator-a",
        runner=commands,
        clock=lambda: next(times),
    )
    first = manager.prove_quiescent()
    second = manager.prove_quiescent()
    assert type(first) is OCIQuiescenceReceipt
    assert (first.sequence, second.sequence) == (1, 2)
    assert first.digest != second.digest

    lease = manager.register(lease_id="lease-1", container_name="container-1")
    with pytest.raises(OCIProcessError, match="not quiescent"):
        manager.prove_quiescent()
    manager.release(lease)
    commands.present.add("container-1")
    with pytest.raises(OCIProcessError, match="not quiescent"):
        manager.prove_quiescent()


def test_legacy_quiescence_digest_is_byte_stable() -> None:
    receipt = OCIQuiescenceReceipt(
        schema="optima.oci-quiescence.v1",
        executor_id="validator-a",
        manager_instance_id="1" * 32,
        namespace_digest="2" * 64,
        sequence=1,
        observed_monotonic_s=10.0,
        lease_records=(),
        resource_entries=(),
        container_ids=(),
    )
    assert receipt.digest == (
        "3a1f289b3087b41503bf160267cb7c14b6ef7d626fdb20e0838db097a7770d06"
    )


def test_transaction_lock_reenters_for_one_controller_thread(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    assert manager.transaction_lock.acquire(blocking=False)
    assert manager.transaction_lock.acquire(blocking=False)
    manager.transaction_lock.release()
    manager.transaction_lock.release()


def test_executor_namespace_has_one_live_manager_owner(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    with pytest.raises(OCIProcessError, match="already owned"):
        _manager(tmp_path)
    manager.close()
    replacement = _manager(tmp_path)
    assert replacement.namespace_digest == manager.namespace_digest
    assert replacement.manager_instance_id != manager.manager_instance_id


def test_quiescence_rejects_malformed_label_listing(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    manager.runner = lambda *_args, **_kwargs: CommandResult(0, b"not-a-container\n", b"")
    with pytest.raises(OCIProcessError, match="malformed"):
        manager.prove_quiescent()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lease_id": "../bad", "container_name": "ok"},
        {"lease_id": "ok", "container_name": "UPPER"},
        {"lease_id": "ok", "container_name": "ok", "stage_relpaths": ("../escape",)},
        {"lease_id": "ok", "container_name": "ok", "mount_relpaths": ("/absolute",)},
        {
            "lease_id": "ok",
            "container_name": "ok",
            "mount_relpaths": ("same",),
            "stage_relpaths": ("same",),
        },
    ],
)
def test_register_rejects_noncanonical_identity_and_resources(tmp_path: Path, kwargs) -> None:
    with pytest.raises(OCIProcessError):
        _manager(tmp_path).register(**kwargs)


def test_run_uses_no_shell_new_session_and_proves_absence(tmp_path: Path, monkeypatch) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    created = []

    def popen(argv, **kwargs):
        proc = FakeProcess(argv, **kwargs)
        created.append(proc)
        return proc

    times = iter((10.0, 12.5))
    monkeypatch.setattr(process_mod.subprocess, "Popen", popen)
    monkeypatch.setattr(manager, "clock", lambda: next(times))
    argv = (*lease.run_prefix(manager.docker_binary), "--network=none", "image@sha256:x")
    result = manager.run(lease, argv, timeout_s=9, stdin_bytes=b"request")

    assert result.returncode == 0 and result.elapsed_seconds == 2.5
    assert created[0].input == b"request"
    assert created[0].kwargs["shell"] is False
    assert created[0].kwargs["start_new_session"] is True
    assert any(row[1:3] == ("container", "ls") for row in commands.rows)
    assert lease.record_path.exists(), "caller retains staged state until publication"


def test_run_continuously_drains_and_retains_only_bounded_stderr_tail(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    real_popen = subprocess.Popen
    marker = b"PREBUILD-TRACEBACK-TAIL"
    script = (
        "import sys; "
        f"sys.stderr.buffer.write(b'A' * {ATTACHED_STDERR_MAX_BYTES * 4}); "
        f"sys.stderr.buffer.write({marker!r}); sys.stderr.buffer.flush(); "
        "raise SystemExit(7)"
    )

    def spawn(_argv, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(process_mod.subprocess, "Popen", spawn)
    result = manager.run(
        lease,
        (*lease.run_prefix(manager.docker_binary), "image@sha256:x"),
        timeout_s=5.0,
    )
    diagnostic = result.stderr_diagnostic
    assert result.returncode == 7 and diagnostic is not None
    assert len(diagnostic.stderr_tail) == ATTACHED_STDERR_MAX_BYTES
    assert diagnostic.stderr_truncated and diagnostic.capture_complete
    assert diagnostic.client_returncode == 7
    assert diagnostic.stderr_tail.endswith(marker)
    artifact = diagnostic.artifact
    assert type(artifact) is OCIStderrArtifactReceipt
    expected = b"A" * (ATTACHED_STDERR_MAX_BYTES * 4) + marker
    assert diagnostic.stream_bytes == len(expected)
    assert diagnostic.stream_sha256 == hashlib.sha256(expected).hexdigest()
    assert artifact.stream_bytes == len(expected)
    assert artifact.stream_sha256 == hashlib.sha256(expected).hexdigest()
    assert artifact.artifact_bytes == len(expected)
    assert not artifact.truncated
    assert manager.reopen_stderr_artifact(artifact) == artifact.artifact_path
    assert artifact.artifact_path.read_bytes() == expected
    assert artifact.receipt_path.read_bytes() == artifact.receipt_bytes
    assert artifact.artifact_path.stat().st_mode & 0o777 == 0o600
    assert artifact.receipt_path.stat().st_mode & 0o777 == 0o600
    assert (artifact.owner_uid, artifact.owner_gid) == (os.geteuid(), os.getegid())


def test_failure_artifact_retains_earliest_rank_error_and_caps_disk_bytes(
    tmp_path: Path, monkeypatch
) -> None:
    cap = 4096
    monkeypatch.setattr(process_mod, "ATTACHED_STDERR_ARTIFACT_MAX_BYTES", cap)
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    real_popen = subprocess.Popen
    first = b"INITIATING-TP-RANK-ERROR rank=2\n"
    last = b"FINAL-TEARDOWN-NOISE\n"
    payload = first + b"x" * (cap * 3) + last
    script = (
        "import sys; "
        f"sys.stderr.buffer.write({payload!r}); sys.stderr.buffer.flush(); "
        "raise SystemExit(9)"
    )

    def spawn(_argv, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(process_mod.subprocess, "Popen", spawn)
    result = manager.run(
        lease,
        (*lease.run_prefix(manager.docker_binary), "image@sha256:x"),
        timeout_s=5.0,
    )
    diagnostic = result.stderr_diagnostic
    assert diagnostic is not None and diagnostic.artifact is not None
    artifact = diagnostic.artifact
    retained = artifact.artifact_path.read_bytes()
    assert len(retained) == cap == artifact.artifact_bytes
    assert retained == payload[:cap]
    assert retained.startswith(first)
    assert last not in retained
    assert diagnostic.stderr_tail.endswith(last)
    assert artifact.truncated
    assert artifact.stream_bytes == len(payload)
    assert artifact.stream_sha256 == hashlib.sha256(payload).hexdigest()
    assert artifact.artifact_sha256 == hashlib.sha256(payload[:cap]).hexdigest()
    assert manager.reopen_stderr_artifact(artifact) == artifact.artifact_path

    replacement = tmp_path / "attacker-controlled"
    replacement.write_bytes(payload[:cap])
    artifact.artifact_path.unlink()
    artifact.artifact_path.symlink_to(replacement)
    with pytest.raises(OCIProcessError, match="securely reopen"):
        manager.reopen_stderr_artifact(artifact)


def test_successful_attached_finalize_discards_streamed_stderr_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    real_popen = subprocess.Popen
    script = (
        "import sys,time; "
        "sys.stderr.buffer.write(b'benign startup log'); sys.stderr.buffer.flush(); "
        "sys.stdout.buffer.write(b'R'); sys.stdout.buffer.flush(); time.sleep(60)"
    )

    def spawn(_argv, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(process_mod.subprocess, "Popen", spawn)
    client = manager.spawn_attached(
        lease, (*lease.run_prefix(manager.docker_binary), "image@sha256:x")
    )
    readable, _, _ = select.select([client.stdout], [], [], 5.0)
    assert readable and client.stdout.read(1) == b"R"
    client.finalize()
    assert not tuple(manager.diagnostics_root.iterdir())


def test_stderr_receipt_publication_never_unlinks_a_preexisting_path(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    artifact_path = manager.diagnostics_root / (
        "lease-1." + "c" * 32 + ".stderr"
    )
    receipt_path = artifact_path.with_name(artifact_path.name + ".json")
    receipt_path.write_bytes(b"preexisting")
    receipt_path.chmod(0o600)
    empty = hashlib.sha256(b"").hexdigest()
    receipt = OCIStderrArtifactReceipt(
        process_mod.STDERR_ARTIFACT_SCHEMA,
        manager.executor_id,
        "lease-1",
        artifact_path,
        receipt_path,
        empty,
        empty,
        0,
        0,
        False,
        os.geteuid(),
        os.getegid(),
        0o600,
    )
    with pytest.raises(FileExistsError):
        manager._publish_stderr_receipt(receipt)
    assert receipt_path.read_bytes() == b"preexisting"


def test_run_rejects_argv_without_exact_lease_prefix(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    with pytest.raises(OCIProcessError, match="exact lease"):
        manager.run(lease, ("/usr/bin/docker", "run", "image"), timeout_s=1)


def test_attached_client_spawn_and_normal_finalize_use_manager_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    process = FakeProcess(())
    process.stdin = FakeStream()
    process.stdout = FakeStream()
    process.stderr = None
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: process)
    events = []
    original_remove = manager.force_remove_container

    def remove(observed_lease):
        events.append("remove")
        return original_remove(observed_lease)

    def terminate(observed_process):
        events.append("terminate")
        observed_process.terminate()
        observed_process.wait(timeout=10)

    monkeypatch.setattr(manager, "force_remove_container", remove)
    monkeypatch.setattr(manager, "_terminate_client", terminate)
    argv = (*lease.run_prefix(manager.docker_binary), "--network=none", "image@sha256:x")

    client = manager.spawn_attached(lease, argv)
    assert client.stdin is process.stdin and client.stdout is process.stdout
    commands.present.add("container-1")
    client.finalize()

    assert events == ["remove", "terminate", "remove"]
    assert client.closed and process.terminated
    assert process.stdin.closed and process.stdout.closed
    assert "container-1" not in commands.present
    # Teardown is idempotent for a defensive finally: abort after finalize cannot
    # revive or remove a different resource.
    client.abort()
    assert events == ["remove", "terminate", "remove"]


def test_attached_spawn_uses_pipes_no_shell_and_new_process_group(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    created = []

    def popen(argv, **kwargs):
        process = FakeProcess(argv, **kwargs)
        process.stdin = FakeStream()
        process.stdout = FakeStream()
        process.stderr = None
        created.append(process)
        return process

    monkeypatch.setattr(process_mod.subprocess, "Popen", popen)
    argv = (*lease.run_prefix(manager.docker_binary), "image@sha256:x")
    client = manager.spawn_attached(lease, argv)

    assert created[0].kwargs == {
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "bufsize": 0,
        "close_fds": True,
        "start_new_session": True,
        "shell": False,
    }
    client.abort()


def test_attached_stderr_is_continuously_drained_and_tail_bounded(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    real_popen = subprocess.Popen
    marker = b"TRACEBACK-TAIL"
    script = (
        "import sys,time; "
        f"sys.stderr.buffer.write(b'A' * {ATTACHED_STDERR_MAX_BYTES * 4}); "
        f"sys.stderr.buffer.write({marker!r}); sys.stderr.buffer.flush(); "
        "sys.stdout.buffer.write(b'R'); sys.stdout.buffer.flush(); time.sleep(60)"
    )

    def spawn(_argv, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(process_mod.subprocess, "Popen", spawn)
    client = manager.spawn_attached(
        lease, (*lease.run_prefix(manager.docker_binary), "image@sha256:x")
    )
    try:
        readable, _, _ = select.select([client.stdout], [], [], 5.0)
        assert readable, "stderr backpressure prevented the child readiness marker"
        assert client.stdout.read(1) == b"R"
        for _ in range(200):
            diagnostic = client.stderr_diagnostic()
            if diagnostic.stderr_tail.endswith(marker):
                break
            time.sleep(0.01)
        else:
            pytest.fail("bounded stderr drain did not preserve the emitted tail")
        assert len(diagnostic.stderr_tail) == ATTACHED_STDERR_MAX_BYTES
        assert diagnostic.stderr_truncated
        assert diagnostic.stderr_tail.endswith(marker)
        assert not diagnostic.capture_complete
    finally:
        client.abort()

    final = client.stderr_diagnostic()
    assert final.capture_complete
    assert final.client_returncode is not None
    assert final.stderr_tail == diagnostic.stderr_tail
    assert final.stderr_sha256 == diagnostic.stderr_sha256


def test_attached_diagnostic_rejects_overflow_and_renders_a_bounded_safe_excerpt() -> None:
    with pytest.raises(OCIProcessError, match="diagnostic"):
        OCIAttachedDiagnostic(
            b"x" * (ATTACHED_STDERR_MAX_BYTES + 1), True, True
        )

    diagnostic = OCIAttachedDiagnostic(
        b"\x1b\x00" * (ATTACHED_STDERR_MAX_BYTES // 2), True, True
    )
    rendered = diagnostic.summary
    assert len(rendered.encode("utf-8")) <= 10 << 10
    assert "\x1b" not in rendered and "\x00" not in rendered
    assert "\\x1b" in rendered and "\\x00" in rendered


def test_attached_abort_rechecks_container_after_client_death(
    tmp_path: Path, monkeypatch
) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    process = FakeProcess(())
    process.stdin = FakeStream()
    process.stdout = FakeStream()
    process.stderr = None
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: process)
    client = manager.spawn_attached(
        lease, (*lease.run_prefix(manager.docker_binary), "image@sha256:x")
    )

    def terminate_then_late_create(observed_process):
        observed_process.terminate()
        commands.present.add("container-1")

    monkeypatch.setattr(manager, "_terminate_client", terminate_then_late_create)
    client.abort()

    assert client.closed
    assert "container-1" not in commands.present
    assert any(row[1:3] == ("rm", "--force") for row in commands.rows)


def test_attached_finalize_never_signals_an_already_reaped_process_group(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    process = FakeProcess(())
    process.stdin = FakeStream()
    process.stdout = FakeStream()
    process.stderr = None
    process.returncode = 0
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: process)
    client = manager.spawn_attached(
        lease, (*lease.run_prefix(manager.docker_binary), "image@sha256:x")
    )
    monkeypatch.setattr(
        process_mod.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(AssertionError("reaped PID was signalled")),
    )

    client.finalize()

    assert client.closed
    assert not process.terminated and not process.killed


@pytest.mark.skipif(sys.platform != "linux", reason="process-group residue proof uses /proc")
def test_attached_finalize_kills_descendant_after_unreaped_leader_exit(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    child_pid_path = tmp_path / "descendant.pid"
    script = (
        "import pathlib,subprocess; "
        f"p=subprocess.Popen(['sleep','60']); pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid))"
    )
    real_popen = subprocess.Popen

    def spawn(_argv, **kwargs):
        return real_popen([sys.executable, "-c", script], **kwargs)

    monkeypatch.setattr(process_mod.subprocess, "Popen", spawn)
    client = manager.spawn_attached(
        lease, (*lease.run_prefix(manager.docker_binary), "image@sha256:x")
    )
    for _ in range(200):
        if child_pid_path.exists():
            break
        time.sleep(0.01)
    assert child_pid_path.exists()
    descendant = int(child_pid_path.read_text())

    # The short-lived leader may already be a zombie, but the manager has not
    # reaped it. Its process group still identifies the surviving descendant.
    client.finalize()
    for _ in range(200):
        try:
            os.kill(descendant, 0)
        except ProcessLookupError:
            break
        try:
            state = Path(f"/proc/{descendant}/stat").read_text().split()[2]
        except OSError:
            break
        if state == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail(f"attached cleanup left descendant process {descendant} alive")


def test_attached_cleanup_failure_still_terminates_and_runs_second_proof(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    process = FakeProcess(())
    process.stdin = FakeStream()
    process.stdout = FakeStream()
    process.stderr = None
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: process)
    client = manager.spawn_attached(
        lease, (*lease.run_prefix(manager.docker_binary), "image@sha256:x")
    )
    proofs = []

    def remove(_lease):
        proofs.append("proof")
        if len(proofs) == 1:
            raise OCIProcessError("first absence proof unavailable")

    monkeypatch.setattr(manager, "force_remove_container", remove)
    monkeypatch.setattr(process_mod.os, "killpg", lambda *_args: (_ for _ in ()).throw(OSError()))
    with pytest.raises(OCIProcessError, match="could not prove"):
        client.abort()

    assert proofs == ["proof", "proof"]
    assert process.terminated
    assert not client.closed


def test_attached_spawn_rejects_wrong_prefix_and_occupied_name(tmp_path: Path) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    with pytest.raises(OCIProcessError, match="exact lease"):
        manager.spawn_attached(lease, ("/usr/bin/docker", "run", "image"))

    commands.present.add("container-1")
    with pytest.raises(OCIProcessError, match="already occupied"):
        manager.spawn_attached(
            lease, (*lease.run_prefix(manager.docker_binary), "image@sha256:x")
        )


def test_timeout_force_removes_container_and_terminates_client(tmp_path: Path, monkeypatch) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    proc = FakeProcess(())

    def communicate(*, input, timeout):
        commands.present.add("container-1")
        raise subprocess.TimeoutExpired("docker", timeout)

    proc.communicate = communicate
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(process_mod.os, "killpg", lambda *_args: (_ for _ in ()).throw(OSError()))

    with pytest.raises(OCIProcessTimeout) as raised:
        manager.run(
            lease,
            (*lease.run_prefix(manager.docker_binary), "image@sha256:x"),
            timeout_s=0.1,
        )
    assert proc.terminated
    assert raised.value.diagnostic is not None
    assert raised.value.diagnostic.artifact is not None
    assert manager.reopen_stderr_artifact(
        raised.value.diagnostic.artifact
    ) == raised.value.diagnostic.artifact.artifact_path
    assert any(row[1:3] == ("rm", "--force") for row in commands.rows)


def test_timeout_rechecks_after_client_death_closes_late_create_race(
    tmp_path: Path, monkeypatch
) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    proc = FakeProcess(())

    def timeout(*, input, timeout):
        raise subprocess.TimeoutExpired("docker", timeout)

    def terminate_then_late_create(_process):
        commands.present.add("container-1")

    proc.communicate = timeout
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(manager, "_terminate_client", terminate_then_late_create)

    with pytest.raises(OCIProcessTimeout):
        manager.run(
            lease,
            (*lease.run_prefix(manager.docker_binary), "image@sha256:x"),
            timeout_s=0.1,
        )
    assert "container-1" not in commands.present
    assert any(row[1:3] == ("rm", "--force") for row in commands.rows)


def test_timeout_still_terminates_client_when_absence_proof_fails(
    tmp_path: Path, monkeypatch
) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    proc = FakeProcess(())
    def timeout(*, input, timeout):
        commands.present.add("container-1")
        raise subprocess.TimeoutExpired("docker", timeout)

    proc.communicate = timeout
    monkeypatch.setattr(process_mod.subprocess, "Popen", lambda *args, **kwargs: proc)
    monkeypatch.setattr(process_mod.os, "killpg", lambda *_args: (_ for _ in ()).throw(OSError()))
    monkeypatch.setattr(
        manager,
        "force_remove_container",
        lambda _lease: (_ for _ in ()).throw(OCIProcessError("absence unavailable")),
    )
    with pytest.raises(OCIProcessError, match="could not prove"):
        manager.run(
            lease,
            (*lease.run_prefix(manager.docker_binary), "image@sha256:x"),
            timeout_s=0.1,
        )
    assert proc.terminated


def test_absence_listing_is_authoritative(tmp_path: Path) -> None:
    commands = Commands()
    commands.present.add("container-1")
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    commands.present.add("container-1")
    # Simulate a daemon that claims rm success but leaves the container present.
    original = commands.__call__

    def stuck(argv, *, timeout_s, max_output_bytes):
        if tuple(argv)[1:3] == ("rm", "--force"):
            commands.rows.append(tuple(argv))
            return CommandResult(0, b"", b"")
        return original(argv, timeout_s=timeout_s, max_output_bytes=max_output_bytes)

    manager.runner = stuck
    with pytest.raises(OCIProcessError, match="still exists"):
        manager.force_remove_container(lease)


def test_cleanup_refuses_same_name_container_with_wrong_lease_labels(tmp_path: Path) -> None:
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    commands.present.add("container-1")
    commands.labels["container-1"] = ("another-executor", "another-lease")
    with pytest.raises(OCIProcessError, match="exact lease labels"):
        manager.force_remove_container(lease)
    assert "container-1" in commands.present
    assert not any(row[1:3] == ("rm", "--force") for row in commands.rows)


def test_release_removes_only_lease_owned_stage_and_record(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(
        lease_id="lease-1",
        container_name="container-1",
        stage_relpaths=("stage",),
    )
    lease.stage_paths[0].mkdir(parents=True)
    (lease.stage_paths[0] / "artifact").write_bytes(b"x")
    unrelated = tmp_path / "unrelated"
    unrelated.write_bytes(b"keep")
    manager.release(lease)
    assert not lease.resource_root.exists()
    assert not lease.record_path.exists()
    assert unrelated.read_bytes() == b"keep"


def test_mount_tmpfs_is_quota_bounded_lease_owned_and_non_executable(
    tmp_path: Path, monkeypatch
) -> None:
    if os.getuid() == 0:
        pytest.skip("mocked tmpfs ownership needs a non-root test user")
    commands = Commands()
    manager = _manager(tmp_path, commands)
    lease = manager.register(
        lease_id="lease-1",
        container_name="container-1",
        mount_relpaths=("native-stage",),
    )
    monkeypatch.setattr(process_mod.os.path, "ismount", lambda path: Path(path) == lease.mount_paths[0])
    selected = manager.mount_tmpfs(
        lease,
        lease.mount_paths[0],
        size_bytes=16 << 20,
        inode_limit=4_096,
        uid=max(1, os.getuid()),
        gid=max(1, os.getgid()),
        executable=False,
    )
    assert selected == lease.mount_paths[0]
    mount = commands.rows[-1]
    assert mount[:4] == ("/usr/bin/mount", "-t", "tmpfs", "-o")
    options = set(mount[4].split(","))
    assert {"rw", "nosuid", "nodev", "noexec", "size=16777216", "nr_inodes=4096"} <= options
    assert mount[-1] == str(lease.mount_paths[0])


def test_mount_tmpfs_rejects_undeclared_path_and_invalid_bounds(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(
        lease_id="lease-1",
        container_name="container-1",
        mount_relpaths=("declared",),
    )
    with pytest.raises(OCIProcessError, match="not declared"):
        manager.mount_tmpfs(
            lease,
            lease.resource_root / "other",
            size_bytes=16 << 20,
            inode_limit=4_096,
            uid=max(1, os.getuid()),
            gid=max(1, os.getgid()),
        )
    with pytest.raises(OCIProcessError, match="outside"):
        manager.mount_tmpfs(
            lease,
            lease.mount_paths[0],
            size_bytes=1,
            inode_limit=4_096,
            uid=max(1, os.getuid()),
            gid=max(1, os.getgid()),
        )


def test_fresh_manager_recovers_only_its_stale_leases(tmp_path: Path) -> None:
    commands = Commands()
    own = _manager(tmp_path, commands)
    stale = own.register(lease_id="stale", container_name="container-stale")
    active = own.register(lease_id="active", container_name="container-active")
    other = OCIProcessManager(
        docker_binary="/usr/bin/docker",
        recovery_root=tmp_path / "recovery",
        executor_id="validator-b",
        runner=commands,
    ).register(lease_id="other", container_name="container-other")

    own.close()  # simulate controller death; the kernel releases its namespace lock
    restarted = _manager(tmp_path, commands)
    assert restarted.recover_stale(active_lease_ids=("active",)) == ("stale",)
    assert not stale.record_path.exists()
    assert active.record_path.exists()
    assert other.record_path.exists()


def test_recovery_is_idempotent_after_resource_removal_before_record_unlink(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="stale", container_name="container-stale")
    __import__("shutil").rmtree(lease.resource_root)
    assert lease.record_path.exists()

    manager.close()  # simulate controller death before restart recovery
    restarted = _manager(tmp_path)
    assert restarted.recover_stale() == ("stale",)
    assert not lease.record_path.exists()


def test_recovery_reaps_atomic_registration_residue_in_own_namespace(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    orphan = manager.resources_root / "orphan"
    orphan.mkdir()
    temporary = manager.leases_root / ".orphan.1234.tmp"
    temporary.write_text("complete but unpublished")

    assert manager.recover_stale() == ("orphan",)
    assert not orphan.exists() and not temporary.exists()


def test_recovery_unlinks_crash_window_temporary_before_validating_record(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="stale", container_name="container-stale")
    temporary = manager.leases_root / ".stale.crash-window.tmp"
    os.link(lease.record_path, temporary)
    assert lease.record_path.stat().st_nlink == 2

    assert manager.recover_stale() == ("stale",)
    assert not temporary.exists()
    assert not lease.record_path.exists()
    assert not lease.resource_root.exists()


def test_recovery_reaps_bounded_native_publication_copy_residue(
    tmp_path: Path,
) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(
        lease_id="stale-publish",
        container_name="container-stale-publish",
        stage_relpaths=("publication-work",),
    )
    work = lease.stage_paths[0]
    residue = work / (".stage-deadbeefdeadbeef-" + "0" * 32)
    # Construct the exact crash shape: the durable lease exists and a bounded
    # validator copy is incomplete beneath its declared work root.
    residue.mkdir(parents=True)
    (residue / "partial.so").write_bytes(b"partial")

    manager.close()  # simulate controller death before restart recovery
    restarted = _manager(tmp_path)
    assert restarted.recover_stale() == ("stale-publish",)
    assert not lease.resource_root.exists()
    assert not lease.record_path.exists()


def test_corrupt_or_symlinked_recovery_record_fails_closed(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    bad = manager.leases_root / "bad.json"
    bad.write_text("{}")
    with pytest.raises(OCIProcessError, match="schema"):
        manager.recover_stale()
    bad.unlink()
    target = tmp_path / "target"
    target.write_text("{}")
    bad.symlink_to(target)
    with pytest.raises(OCIProcessError, match="regular"):
        manager.recover_stale()
