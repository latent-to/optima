from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import optima.eval.oci_process as process_mod
from optima.eval.oci_process import (
    CommandResult,
    OCIProcessError,
    OCIProcessManager,
    OCIProcessTimeout,
)


CONTAINER_ID = "d" * 64


class Commands:
    def __init__(self) -> None:
        self.rows: list[tuple[str, ...]] = []
        self.present: set[str] = set()
        self.labels: dict[str, tuple[str, str]] = {}

    def __call__(self, argv, *, timeout_s, max_output_bytes):
        row = tuple(argv)
        self.rows.append(row)
        if row[1:3] == ("container", "ls"):
            name = next(value for value in row if value.startswith("name=^/"))[7:-1]
            return CommandResult(0, (CONTAINER_ID + "\n").encode() if name in self.present else b"", b"")
        if row[1:3] == ("container", "inspect"):
            name = next(iter(self.present))
            executor, lease = self.labels.get(name, ("validator-a", "lease-1"))
            payload = {
                "Id": CONTAINER_ID,
                "Name": f"/{name}",
                "Labels": {
                    "optima.executor_id": executor,
                    "optima.lease_id": lease,
                },
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
        self.returncode = 0
        self.input = None
        self.waits = []
        self.terminated = False
        self.killed = False

    def communicate(self, *, input, timeout):
        self.input = input
        self.waits.append(timeout)

    def wait(self, timeout):
        self.waits.append(timeout)
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


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


def test_run_rejects_argv_without_exact_lease_prefix(tmp_path: Path) -> None:
    manager = _manager(tmp_path)
    lease = manager.register(lease_id="lease-1", container_name="container-1")
    with pytest.raises(OCIProcessError, match="exact lease"):
        manager.run(lease, ("/usr/bin/docker", "run", "image"), timeout_s=1)


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

    with pytest.raises(OCIProcessTimeout):
        manager.run(
            lease,
            (*lease.run_prefix(manager.docker_binary), "image@sha256:x"),
            timeout_s=0.1,
        )
    assert proc.terminated
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
