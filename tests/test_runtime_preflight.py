"""Adversarial tests for the candidate-free runtime image preflight."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sysconfig

import pytest

from optima.eval import runtime_preflight as rp
from optima.eval.oci_process import OCIProcessManager
from optima.eval.runtime_preflight import (
    CommandResult,
    RuntimePreflightConfig,
    RuntimePreflightError,
)


IMAGE = "registry.example/optima-runtime@sha256:" + "a" * 64
LOCAL_IMAGE_ID = "sha256:" + "b" * 64
WORKER_DIGEST = "c" * 64
DOCKER = "/usr/bin/docker"
CONTAINER_NAME = "optima-runtime-preflight-" + "1" * 20
CONTAINER_ID = "d" * 64


class Clock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value


class ScriptedRunner:
    def __init__(self, results, *, clock: Clock | None = None, advances=()):
        self.results = list(results)
        self.calls = []
        self.clock = clock
        self.advances = list(advances)

    def __call__(
        self,
        argv,
        *,
        timeout_s,
        max_stdout_bytes,
        max_stderr_bytes,
    ):
        self.calls.append(
            (
                tuple(argv),
                float(timeout_s),
                max_stdout_bytes,
                max_stderr_bytes,
            )
        )
        if self.clock is not None and self.advances:
            self.clock.value += self.advances.pop(0)
        result = self.results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class ControlRunner:
    def __init__(self):
        self.calls = []
        self.present = False
        self.labels = (
            "preflight",
            "0" * 64,
            "preflight-" + "1" * 20,
        )

    def __call__(self, argv, *, timeout_s, max_output_bytes):
        row = tuple(argv)
        self.calls.append(row)
        if row[1:3] == ("container", "ls"):
            return CommandResult(
                0, (CONTAINER_ID + "\n").encode() if self.present else b"", b""
            )
        if row[1:3] == ("container", "inspect"):
            executor, namespace, lease = self.labels
            return CommandResult(
                0,
                json.dumps(
                    {
                        "Id": CONTAINER_ID,
                        "Name": f"/{CONTAINER_NAME}",
                        "Labels": {
                            "optima.executor_id": executor,
                            "optima.namespace_digest": namespace,
                            "optima.lease_id": lease,
                        },
                    }
                ).encode(),
                b"",
            )
        if row[1:3] == ("rm", "--force"):
            self.present = False
        return CommandResult(0, b"", b"")


def _config(**changes) -> RuntimePreflightConfig:
    values = {
        "image": IMAGE,
        "expected_oci_platform": "linux/amd64",
        "expected_python_platform": "linux-x86_64",
        "expected_machine": "x86_64",
        "expected_python_executable": "/usr/local/bin/python3",
        "expected_sglang_version": "0.5.2",
        "expected_worker_distribution": "optima-harness",
        "expected_worker_version": "0.0.1",
        "expected_worker_digest": WORKER_DIGEST,
        "uid": 65532,
        "gid": 65532,
        "docker_binary": DOCKER,
        "timeout_s": 60.0,
    }
    values.update(changes)
    return RuntimePreflightConfig(**values)


def _inspect(
    *,
    repo_digests=None,
    image_id=LOCAL_IMAGE_ID,
    volumes=None,
    os_name="linux",
    architecture="amd64",
    extra=None,
):
    payload = {
        "Id": image_id,
        "RepoDigests": [IMAGE] if repo_digests is None else repo_digests,
        "Volumes": volumes,
        "Os": os_name,
        "Architecture": architecture,
    }
    if extra:
        payload.update(extra)
    return CommandResult(0, json.dumps(payload).encode(), b"")


def _container_payload(
    *,
    sglang="0.5.2",
    worker_distribution="optima-harness",
    worker_version="0.0.1",
    worker_digest=WORKER_DIGEST,
    python_platform="linux-x86_64",
    machine="x86_64",
    extra=None,
):
    payload = {
        "schema": rp.CONTAINER_RECEIPT_SCHEMA,
        "sglang_version": sglang,
        "worker": {
            "distribution": worker_distribution,
            "version": worker_version,
            "digest": worker_digest,
            "file_count": 157,
            "total_bytes": 654321,
        },
        "python": {
            "executable": "/usr/local/bin/python3",
            "implementation": "cpython",
            "version": "3.11.15",
            "abi": "cpython-311-x86_64-linux-gnu",
            "platform": python_platform,
            "machine": machine,
        },
        "packages": {
            "cuda-python": "12.9.0",
            "flashinfer-python": "0.6.12",
            "nvidia-cuda-runtime-cu12": "12.9.79",
            "torch": "2.9.1",
            "triton": "3.5.1",
        },
        "cuda": {
            "cudart_library": "libcudart.so.12",
            "cuda_visible_devices": "",
            "nvidia_visible_devices": "void",
        },
    }
    if extra:
        payload.update(extra)
    return payload


def _container(**changes):
    return CommandResult(0, json.dumps(_container_payload(**changes)).encode(), b"")


def _successful_runner(**container_changes):
    return ScriptedRunner([_inspect(), _container(**container_changes)])


def run_runtime_preflight(
    config, *, runner=None, clock=rp.time.monotonic, process_manager=None
):
    if process_manager is not None:
        return rp.run_runtime_preflight(
            config,
            runner=rp._bounded_argv_runner if runner is None else runner,
            clock=clock,
            process_manager=process_manager,
        )
    assert runner is not None
    return rp._run_runtime_preflight_unleased_for_test(
        config, runner=runner, clock=clock
    )


def test_success_binds_image_platform_and_installed_worker_identity():
    runner = _successful_runner()
    receipt = run_runtime_preflight(_config(), runner=runner, clock=Clock())

    expected_platform_digest = rp._platform_digest(
        oci_platform="linux/amd64",
        python_implementation="cpython",
        python_executable="/usr/local/bin/python3",
        python_version="3.11.15",
        python_abi="cpython-311-x86_64-linux-gnu",
        python_platform="linux-x86_64",
        machine="x86_64",
    )
    assert receipt.requested_image == IMAGE
    assert receipt.image_digest == "a" * 64
    assert receipt.local_image_id == LOCAL_IMAGE_ID
    assert receipt.repo_digests == (IMAGE,)
    assert receipt.oci_platform == "linux/amd64"
    assert receipt.platform_digest == expected_platform_digest
    assert receipt.worker_distribution == "optima-harness"
    assert receipt.worker_version == "0.0.1"
    assert receipt.worker_distribution_digest == WORKER_DIGEST
    assert receipt.launch_identity() == {
        "image_digest": "a" * 64,
        "platform_digest": expected_platform_digest,
        "worker_distribution_digest": WORKER_DIGEST,
    }
    assert len(receipt.sha256) == 64
    assert receipt.sha256 == rp.hashlib.sha256(
        receipt.canonical_json.encode("ascii")
    ).hexdigest()
    assert hash(receipt)
    assert json.loads(receipt.canonical_json) == receipt.canonical_payload()


@pytest.mark.parametrize(
    "field,value",
    (
        ("oci_platform", "linux/arm64"),
        ("python_implementation", "pypy"),
        ("python_executable", "/usr/bin/python3"),
        ("python_version", "3.11.16"),
        ("python_abi", "cpython-311-other"),
        ("python_platform", "linux-aarch64"),
        ("machine", "aarch64"),
    ),
)
def test_platform_digest_rotates_every_interpreter_and_abi_input(field, value):
    identity = {
        "oci_platform": "linux/amd64",
        "python_implementation": "cpython",
        "python_executable": "/usr/local/bin/python3",
        "python_version": "3.11.15",
        "python_abi": "cpython-311-x86_64-linux-gnu",
        "python_platform": "linux-x86_64",
        "machine": "x86_64",
    }
    original = rp._platform_digest(**identity)
    identity[field] = value
    assert rp._platform_digest(**identity) != original


def test_exact_inspect_and_mountless_no_gpu_no_capability_container_argv(monkeypatch):
    monkeypatch.setattr(rp.secrets, "token_hex", lambda size: "1" * (size * 2))
    runner = _successful_runner()
    receipt = run_runtime_preflight(_config(), runner=runner, clock=Clock())

    assert runner.calls[0][0] == (
        DOCKER,
        "image",
        "inspect",
        rp._INSPECT_FORMAT,
        IMAGE,
    )
    run_argv = runner.calls[1][0]
    assert run_argv == (
        DOCKER,
        "run",
        "--rm",
        "--pull=never",
        "--platform=linux/amd64",
        "--network=none",
        "--read-only",
        "--runtime=runc",
        "--ipc=none",
        f"--name={CONTAINER_NAME}",
        "--stop-timeout=1",
        "--no-healthcheck",
        "--user=65532:65532",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges=true",
        "--security-opt=seccomp=builtin",
        "--pids-limit=32",
        "--memory=512m",
        "--memory-swap=512m",
        "--cpus=1.0",
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=64m",
        "--workdir=/tmp",
        "--env=NVIDIA_VISIBLE_DEVICES=void",
        "--env=CUDA_VISIBLE_DEVICES=",
        "--log-driver=none",
        "--entrypoint=/usr/local/bin/python3",
        LOCAL_IMAGE_ID,
        "-I",
        "-S",
        "-c",
        rp._CONTAINER_SCRIPT,
    )
    assert not any(
        arg == "--gpus"
        or arg.startswith(("--gpus=", "--mount=", "--volume=", "--cap-add="))
        or arg in {"-v", "--mount", "--volume", "--cap-add"}
        for arg in run_argv
    )
    assert receipt.security_argv_sha256 == rp.hashlib.sha256(
        json.dumps(run_argv, separators=(",", ":")).encode()
    ).hexdigest()


def test_production_preflight_is_lease_owned_and_released(tmp_path, monkeypatch):
    monkeypatch.setattr(rp.secrets, "token_hex", lambda size: "1" * (size * 2))
    controls = ControlRunner()
    manager = OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=tmp_path / "recovery",
        executor_id="preflight",
        runner=controls,
    )
    controls.labels = (
        "preflight",
        manager.namespace_digest,
        "preflight-" + "1" * 20,
    )
    runner = _successful_runner()

    receipt = run_runtime_preflight(
        _config(), runner=runner, clock=Clock(), process_manager=manager
    )

    run_argv = runner.calls[1][0]
    lease_id = "preflight-" + "1" * 20
    assert run_argv[:8] == (
        DOCKER,
        "run",
        f"--name={CONTAINER_NAME}",
        f"--cidfile={manager.resources_root / lease_id / 'container.cid'}",
        "--label=optima.executor_id=preflight",
        f"--label=optima.namespace_digest={manager.namespace_digest}",
        f"--label=optima.lease_id={lease_id}",
        "--rm",
    )
    assert receipt.security_argv_sha256 == rp.hashlib.sha256(
        json.dumps(run_argv, separators=(",", ":")).encode()
    ).hexdigest()
    assert not list(manager.leases_root.iterdir())
    assert not list(manager.resources_root.iterdir())
    assert sum(row[1:3] == ("container", "ls") for row in controls.calls) >= 2


def test_production_preflight_refuses_unleased_default_runner():
    with pytest.raises(RuntimePreflightError, match="OCIProcessManager lease"):
        rp.run_runtime_preflight(_config())


def test_leased_preflight_timeout_removes_exact_labeled_container(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(rp.secrets, "token_hex", lambda size: "1" * (size * 2))
    controls = ControlRunner()
    manager = OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=tmp_path / "recovery",
        executor_id="preflight",
        runner=controls,
    )
    controls.labels = (
        "preflight",
        manager.namespace_digest,
        "preflight-" + "1" * 20,
    )

    class TimeoutRunner(ScriptedRunner):
        def __call__(self, argv, **kwargs):
            if len(self.calls) == 1:
                controls.present = True
            return super().__call__(argv, **kwargs)

    runner = TimeoutRunner([_inspect(), subprocess.TimeoutExpired((DOCKER, "run"), 1)])
    with pytest.raises(RuntimePreflightError, match="timed out"):
        run_runtime_preflight(
            _config(), runner=runner, clock=Clock(), process_manager=manager
        )
    assert not controls.present
    assert any(row[1:3] == ("rm", "--force") for row in controls.calls)
    assert not list(manager.leases_root.iterdir())
    assert not list(manager.resources_root.iterdir())


def test_fixed_probe_uses_metadata_only_and_contains_no_package_imports():
    tree = ast.parse(rp._CONTAINER_SCRIPT)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert imported.isdisjoint({"optima", "torch", "sglang", "triton", "flashinfer"})
    assert "importlib.metadata.distributions" in rp._CONTAINER_SCRIPT
    assert "dist.locate_file" in rp._CONTAINER_SCRIPT
    assert "candidate" not in rp._CONTAINER_SCRIPT.lower()
    assert "sys.path.append" in rp._CONTAINER_SCRIPT


def test_fixed_probe_hashes_installed_worker_without_importing_it(
    tmp_path, monkeypatch, capsys
):
    site = tmp_path / "site-packages"
    module = site / "optima" / "__init__.py"
    metadata_dir = site / "optima_harness-0.0.1.dist-info"
    metadata = metadata_dir / "METADATA"
    record = metadata_dir / "RECORD"
    module.parent.mkdir(parents=True)
    metadata_dir.mkdir()
    module.write_bytes(b'raise RuntimeError("must never import worker")\n')
    metadata.write_bytes(b"Metadata-Version: 2.1\nName: optima-harness\nVersion: 0.0.1\n")
    record.write_text(
        "optima/__init__.py,,\n"
        "optima_harness-0.0.1.dist-info/METADATA,,\n"
        "optima_harness-0.0.1.dist-info/RECORD,,\n"
        "../../../bin/optima,,\n"
    )
    original_get_path = sysconfig.get_path
    monkeypatch.setattr(
        sysconfig,
        "get_path",
        lambda name: str(site)
        if name in {"purelib", "platlib"}
        else original_get_path(name),
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", "void")

    exec(compile(rp._CONTAINER_SCRIPT, "<preflight>", "exec"), {})
    payload = json.loads(capsys.readouterr().out)
    rows = []
    for path in (module, metadata, record):
        relative = path.relative_to(site).as_posix()
        content = path.read_bytes()
        rows.append([relative, len(content), hashlib.sha256(content).hexdigest()])
    identity = {
        "schema": rp.WORKER_DIGEST_SCHEMA,
        "distribution": "optima-harness",
        "version": "0.0.1",
        "files": sorted(rows),
    }
    expected = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert payload["worker"] == {
        "distribution": "optima-harness",
        "version": "0.0.1",
        "digest": expected,
        "file_count": 3,
        "total_bytes": sum(len(path.read_bytes()) for path in (module, metadata, record)),
    }


def test_each_preflight_uses_a_unique_container_name(monkeypatch):
    tokens = iter(("1" * 20, "2" * 20))
    monkeypatch.setattr(rp.secrets, "token_hex", lambda _size: next(tokens))
    runners = (_successful_runner(), _successful_runner())

    for runner in runners:
        run_runtime_preflight(_config(), runner=runner, clock=Clock())

    names = [
        next(arg for arg in runner.calls[1][0] if arg.startswith("--name="))
        for runner in runners
    ]
    assert names == [
        "--name=optima-runtime-preflight-" + "1" * 20,
        "--name=optima-runtime-preflight-" + "2" * 20,
    ]


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"image": "registry.example/runtime:latest"}, "name@sha256"),
        ({"image": "registry.example/runtime@sha256:" + "A" * 64}, "name@sha256"),
        ({"expected_oci_platform": "linux"}, "OCI platform"),
        ({"expected_python_platform": ""}, "expected_python_platform"),
        ({"expected_machine": "bad\nvalue"}, "expected_machine"),
        ({"expected_worker_distribution": "other"}, "optima-harness"),
        ({"expected_worker_version": "bad;version"}, "worker_version"),
        ({"expected_worker_digest": "A" * 64}, "lowercase sha256"),
        ({"expected_worker_digest": "c" * 63}, "lowercase sha256"),
        ({"docker_binary": "docker"}, "absolute normalized"),
        ({"docker_binary": "//usr/bin/docker"}, "absolute normalized"),
        ({"docker_binary": "/usr/bin/../bin/docker"}, "absolute normalized"),
        ({"docker_binary": "/usr/bin/docker;rm"}, "absolute normalized"),
        ({"uid": 0}, "nonzero"),
        ({"uid": True}, "nonzero"),
        ({"gid": 0}, "nonzero"),
        ({"expected_sglang_version": "0.5.2;bad"}, "sglang_version"),
    ],
)
def test_config_rejects_mutable_or_noncanonical_identity(changes, match):
    with pytest.raises(RuntimePreflightError, match=match):
        _config(**changes)


def test_repo_digest_must_bind_requested_image_to_local_id():
    other = "registry.example/runtime@sha256:" + "d" * 64
    runner = ScriptedRunner([_inspect(repo_digests=[other])])

    with pytest.raises(RuntimePreflightError, match="not bound"):
        run_runtime_preflight(_config(), runner=runner, clock=Clock())
    assert len(runner.calls) == 1


@pytest.mark.parametrize(
    "inspect_result, match",
    [
        (CommandResult(0, b"not-json", b""), "malformed JSON"),
        (_inspect(extra={"RepoTags": []}), "keys/type mismatch"),
        (_inspect(image_id="sha256:short"), "local image ID"),
        (_inspect(repo_digests=[IMAGE, IMAGE]), "invalid RepoDigests"),
        (_inspect(volumes={"/candidate-state": {}}), "Dockerfile volumes"),
        (_inspect(architecture="arm64"), "OCI platform mismatch"),
    ],
)
def test_image_inspect_is_strict_and_platform_bound(inspect_result, match):
    runner = ScriptedRunner([inspect_result])
    with pytest.raises(RuntimePreflightError, match=match):
        run_runtime_preflight(_config(), runner=runner, clock=Clock())


@pytest.mark.parametrize(
    "raw, match",
    [
        (b"not-json", "malformed JSON"),
        (json.dumps(_container_payload()).encode() + b"\nextra", "malformed JSON"),
        (
            json.dumps(_container_payload(extra={"candidate": "forbidden"})).encode(),
            "keys/type mismatch",
        ),
        (
            b'{"schema":"x","schema":"y"}',
            "duplicate key",
        ),
    ],
)
def test_container_receipt_rejects_malformed_extra_or_duplicate_output(raw, match):
    runner = ScriptedRunner([_inspect(), CommandResult(0, raw, b"")])
    with pytest.raises(RuntimePreflightError, match=match):
        run_runtime_preflight(_config(), runner=runner, clock=Clock())


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"sglang": "0.5.3"}, "sglang mismatch"),
        ({"worker_distribution": "other"}, "worker distribution mismatch"),
        ({"worker_version": "0.0.2"}, "worker version mismatch"),
        ({"worker_digest": "d" * 64}, "worker distribution digest mismatch"),
        ({"worker_digest": "short"}, "worker distribution digest is invalid"),
        ({"python_platform": "linux-aarch64"}, "Python platform mismatch"),
        ({"machine": "aarch64"}, "machine mismatch"),
    ],
)
def test_runtime_identity_mismatch_is_validator_fault(changes, match):
    runner = _successful_runner(**changes)
    with pytest.raises(RuntimePreflightError, match=match) as caught:
        run_runtime_preflight(_config(), runner=runner, clock=Clock())
    assert caught.value.validator_fault is True
    assert caught.value.retryable is False


@pytest.mark.parametrize(
    "worker_change, match",
    [
        ({"file_count": 0}, "inventory is empty"),
        ({"file_count": True}, "bounded integer"),
        ({"file_count": 4097}, "bounded integer"),
        ({"total_bytes": 0}, "inventory is empty"),
        ({"total_bytes": 64 * 1024 * 1024 + 1}, "bounded integer"),
        ({"extra": 1}, "keys/type mismatch"),
    ],
)
def test_worker_receipt_inventory_is_exact_and_bounded(worker_change, match):
    payload = _container_payload()
    payload["worker"].update(worker_change)
    runner = ScriptedRunner(
        [_inspect(), CommandResult(0, json.dumps(payload).encode(), b"")]
    )
    with pytest.raises(RuntimePreflightError, match=match):
        run_runtime_preflight(_config(), runner=runner, clock=Clock())


def test_timeout_uses_one_absolute_deadline_then_bounded_cleanup(monkeypatch):
    monkeypatch.setattr(rp.secrets, "token_hex", lambda size: "1" * (size * 2))
    clock = Clock()
    runner = ScriptedRunner(
        [
            _inspect(),
            subprocess.TimeoutExpired((DOCKER, "run"), 1.0),
            CommandResult(0, b"removed\n", b""),
        ],
        clock=clock,
        advances=(59.0, 0.0),
    )
    with pytest.raises(RuntimePreflightError, match="timed out") as caught:
        run_runtime_preflight(_config(), runner=runner, clock=clock)
    assert runner.calls[0][1] == pytest.approx(60.0)
    assert runner.calls[1][1] == pytest.approx(1.0)
    assert runner.calls[2][0] == (
        DOCKER,
        "rm",
        "--force",
        "--volumes",
        CONTAINER_NAME,
    )
    assert runner.calls[2][1] == pytest.approx(5.0)
    assert caught.value.validator_fault is True


def test_failed_probe_with_unconfirmed_cleanup_is_terminal(monkeypatch):
    monkeypatch.setattr(rp.secrets, "token_hex", lambda size: "1" * (size * 2))
    runner = ScriptedRunner(
        [
            _inspect(),
            CommandResult(2, b"", b"probe failed"),
            CommandResult(3, b"", b"cleanup failed"),
        ]
    )
    with pytest.raises(RuntimePreflightError, match="cleanup could not be confirmed"):
        run_runtime_preflight(_config(), runner=runner, clock=Clock())


def test_nonzero_stderr_and_runner_output_limit_fail_closed():
    with pytest.raises(RuntimePreflightError, match="exited 2"):
        run_runtime_preflight(
            _config(),
            runner=ScriptedRunner([CommandResult(2, b"", b"daemon unavailable")]),
            clock=Clock(),
        )
    noisy = ScriptedRunner(
        [
            _inspect(),
            CommandResult(0, json.dumps(_container_payload()).encode(), b"warning"),
            CommandResult(0, b"removed\n", b""),
        ]
    )
    with pytest.raises(RuntimePreflightError, match="unexpected stderr"):
        run_runtime_preflight(_config(), runner=noisy, clock=Clock())

    oversized = ScriptedRunner(
        [CommandResult(0, b"x" * (rp.MAX_INSPECT_STDOUT_BYTES + 1), b"")]
    )
    with pytest.raises(RuntimePreflightError, match="output bounds"):
        run_runtime_preflight(_config(), runner=oversized, clock=Clock())


def test_default_runner_invokes_subprocess_without_shell_or_stdin(monkeypatch):
    captured = {}

    def refuse(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        raise OSError("test stop")

    monkeypatch.setattr(rp.subprocess, "Popen", refuse)
    with pytest.raises(RuntimePreflightError, match="cannot execute"):
        rp._bounded_argv_runner(
            (DOCKER, "version"),
            timeout_s=1.0,
            max_stdout_bytes=16,
            max_stderr_bytes=16,
        )
    assert captured["argv"] == [DOCKER, "version"]
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["stdin"] is subprocess.DEVNULL
    assert captured["kwargs"]["start_new_session"] is True


def test_injected_clock_and_runner_types_fail_as_validator_faults():
    with pytest.raises(RuntimePreflightError, match="clock") as clock_error:
        run_runtime_preflight(_config(), runner=_successful_runner(), clock=lambda: "bad")
    assert clock_error.value.validator_fault is True

    runner = ScriptedRunner([CommandResult(0, "not-bytes", b"")])
    with pytest.raises(RuntimePreflightError, match="field types") as runner_error:
        run_runtime_preflight(_config(), runner=runner, clock=Clock())
    assert runner_error.value.validator_fault is True


def test_deeply_nested_json_fails_closed_as_validator_fault():
    nested = b"[" * 2000 + b"0" + b"]" * 2000
    runner = ScriptedRunner([CommandResult(0, nested, b"")])
    with pytest.raises(RuntimePreflightError) as caught:
        run_runtime_preflight(_config(), runner=runner, clock=Clock())
    assert caught.value.validator_fault is True
