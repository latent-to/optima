from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from optima.eval import oci_session_worker as worker
from optima.eval.oci_session_protocol import (
    CONTROL_MAGIC,
    EVIDENCE_MAGIC,
    FRAME_HEADER_BYTES,
    MAX_BATCH_REQUEST_BYTES,
    MAX_CONTROL_BYTES,
    MAX_INIT_BYTES,
    EngineSessionConfig,
    RuntimePreflightFacts,
    batch_request,
    frame_message,
    make_init,
    parse_evidence_frame_bytes,
    parse_frame_bytes,
    preflight_accept_message,
    validate_batch_request,
)


def _digest(character: str) -> str:
    return character * 64


def _config() -> EngineSessionConfig:
    return EngineSessionConfig(
        model_path="/optima/input/model",
        dtype="bfloat16",
        deterministic=False,
        attention_backend=None,
        disable_cuda_graph=False,
        mem_fraction_static=0.8,
        log_level="warning",
        max_running_requests=16,
        tp_size=1,
        moe_runner_backend=None,
        disable_custom_all_reduce=False,
        engine_kwargs={},
    )


def _facts(config: EngineSessionConfig, launch: str) -> RuntimePreflightFacts:
    return RuntimePreflightFacts(
        launch_digest=launch,
        runtime_digest=_digest("b"),
        stack_digest=_digest("c"),
        tree_digest=_digest("d"),
        engine_config_digest=config.digest,
        worker_distribution_digest=_digest("e"),
        model_revision_digest=_digest("f"),
        model_manifest_digest=_digest("1"),
        model_content_digest=_digest("2"),
        sglang_version="0.0.0.dev1",
        gpu_architectures=("sm120",),
        topology_digest=_digest("3"),
        loopback_only=True,
        read_only_inputs=True,
        private_writable_cache=True,
    )


def _init_frame(config: EngineSessionConfig, *, session: str, launch: str) -> bytes:
    return frame_message(
        make_init(
            config,
            session_id=session,
            launch_digest=launch,
            expected_engine_config_digest=config.digest,
        ),
        max_bytes=MAX_INIT_BYTES,
    )


def _accept_frame(
    config: EngineSessionConfig, *, session: str, launch: str
) -> bytes:
    return frame_message(
        preflight_accept_message(
            session_id=session,
            launch_digest=launch,
            facts=_facts(config, launch),
        ),
        max_bytes=MAX_CONTROL_BYTES,
    )


def _pipe_with(payload: bytes) -> int:
    read_fd, write_fd = os.pipe()
    try:
        os.write(write_fd, payload)
    finally:
        os.close(write_fd)
    return read_fd


def _read_exact(fd: int, size: int) -> bytes:
    chunks = []
    while sum(map(len, chunks)) < size:
        chunk = os.read(fd, size - sum(map(len, chunks)))
        if not chunk:
            raise AssertionError("pipe ended before a complete test frame")
        chunks.append(chunk)
    return b"".join(chunks)


def _read_frame(fd: int) -> bytes:
    header = _read_exact(fd, FRAME_HEADER_BYTES)
    size = int.from_bytes(header[4:8], "big")
    return header + _read_exact(fd, size)


def _session_fds(payload: bytes) -> tuple[int, int, int]:
    input_fd = _pipe_with(payload)
    output_read, output_write = os.pipe()
    return input_fd, output_read, output_write


def _bind_init(monkeypatch, config: EngineSessionConfig, launch: str) -> None:
    monkeypatch.setenv("OPTIMA_LAUNCH_DIGEST", launch)
    monkeypatch.setenv("OPTIMA_ENGINE_CONFIG_DIGEST", config.digest)


def test_topology_digest_normalizes_nvidia_smi_underlined_header(monkeypatch):
    observed = (
        "\t\x1b[4mGPU0\tGPU1\tNIC0\x1b[0m\n"
        "GPU0\t X \tPIX\tPIX\n"
        "GPU1\tPIX\t X \tPIX\n"
    )
    monkeypatch.setattr(worker, "_run_nvidia_smi", lambda *_args: observed)
    payload = json.dumps(
        {
            "matrix": [["X", "PIX"], ["PIX", "X"]],
            "schema": "optima-gpu-topology-v1",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    assert worker._topology_digest() == hashlib.sha256(payload).hexdigest()


def test_topology_digest_rejects_unknown_escape(monkeypatch):
    monkeypatch.setattr(
        worker,
        "_run_nvidia_smi",
        lambda *_args: "\x1b[31mGPU0\x1b[0m\nGPU0 X\n",
    )
    with pytest.raises(worker.SessionWorkerError, match="unknown escape"):
        worker._topology_digest()


def test_importing_worker_loads_no_torch_sglang_or_candidate(tmp_path):
    script = tmp_path / "probe.py"
    script.write_text(
        "import importlib.abc, sys\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "  def find_spec(self, fullname, path=None, target=None):\n"
        "    if fullname.split('.')[0] in {'torch','sglang','miner_payload'}:\n"
        "      raise RuntimeError('forbidden import: ' + fullname)\n"
        "    return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import optima.eval.oci_session_worker\n"
        "assert not any(n.split('.')[0] in {'torch','sglang','miner_payload'} "
        "for n in sys.modules)\n"
    )
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root)}
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_preflight_frame_is_published_before_engine_candidate_or_native_entry(
    monkeypatch,
):
    config = _config()
    session, launch = "4" * 32, _digest("a")
    _bind_init(monkeypatch, config, launch)
    input_fd, output_read, output_write = _session_fds(
        _init_frame(config, session=session, launch=launch)
        + _accept_frame(config, session=session, launch=launch)
    )
    order: list[str] = []

    def preflight(_config, *, launch_digest):
        order.append("preflight")
        return _facts(config, launch_digest), SimpleNamespace(root="tree")

    @contextlib.contextmanager
    def engine_session(_config, _tree):
        # Candidate/SGLang/native entry is represented by entering this context.
        # The complete preflight frame must already be outside the container.
        first = _read_frame(output_read)
        assert first[:4] == CONTROL_MAGIC
        assert parse_frame_bytes(first, max_bytes=MAX_CONTROL_BYTES)["type"] == "preflight"
        order.append("engine")
        yield worker._EngineHandle(SimpleNamespace(), lambda: None)

    monkeypatch.setattr(worker, "_validate_live_preflight", preflight)
    monkeypatch.setattr(worker, "_engine_session", engine_session)
    try:
        assert worker.run_session(input_fd=input_fd, output_fd=output_write) == 1
        assert order == ["preflight", "engine"]
    finally:
        os.close(input_fd)
        os.close(output_write)
        os.close(output_read)


def test_failed_preflight_never_enters_candidate_native_or_sglang(monkeypatch):
    config = _config()
    session, launch = "5" * 32, _digest("a")
    _bind_init(monkeypatch, config, launch)
    input_fd, output_read, output_write = _session_fds(
        _init_frame(config, session=session, launch=launch)
    )
    entered = {"candidate": False, "native": False, "sglang": False}

    def failed_preflight(*_args, **_kwargs):
        raise worker.SessionWorkerError("attestation mismatch")

    @contextlib.contextmanager
    def forbidden_engine(*_args, **_kwargs):
        entered.update({name: True for name in entered})
        yield  # pragma: no cover

    monkeypatch.setattr(worker, "_validate_live_preflight", failed_preflight)
    monkeypatch.setattr(worker, "_engine_session", forbidden_engine)
    try:
        assert worker.run_session(input_fd=input_fd, output_fd=output_write) == 1
        assert entered == {"candidate": False, "native": False, "sglang": False}
        error = parse_frame_bytes(_read_frame(output_read), max_bytes=MAX_CONTROL_BYTES)
        assert error["type"] == "session_error"
        assert error["stage"] == "preflight"
    finally:
        os.close(input_fd)
        os.close(output_write)
        os.close(output_read)


@pytest.mark.parametrize("acceptance", ["missing", "wrong"])
def test_missing_or_wrong_preflight_accept_never_enters_hostile_engine(
    monkeypatch, acceptance
):
    config = _config()
    session, launch = "c" * 32, _digest("a")
    _bind_init(monkeypatch, config, launch)
    payload = _init_frame(config, session=session, launch=launch)
    if acceptance == "wrong":
        wrong = replace(
            _facts(config, launch), runtime_digest=_digest("4")
        )
        payload += frame_message(
            preflight_accept_message(
                session_id=session, launch_digest=launch, facts=wrong
            ),
            max_bytes=MAX_CONTROL_BYTES,
        )
    input_fd, output_read, output_write = _session_fds(payload)
    entered = []
    monkeypatch.setattr(
        worker,
        "_validate_live_preflight",
        lambda _config, *, launch_digest: (
            _facts(config, launch_digest), SimpleNamespace(root="tree")
        ),
    )

    @contextlib.contextmanager
    def forbidden_engine(*_args, **_kwargs):
        entered.append("candidate-native-sglang")
        yield  # pragma: no cover

    monkeypatch.setattr(worker, "_engine_session", forbidden_engine)
    try:
        assert worker.run_session(input_fd=input_fd, output_fd=output_write) == 1
        assert entered == []
        preflight = parse_frame_bytes(
            _read_frame(output_read), max_bytes=MAX_CONTROL_BYTES
        )
        terminal = parse_frame_bytes(
            _read_frame(output_read), max_bytes=MAX_CONTROL_BYTES
        )
        assert preflight["type"] == "preflight"
        assert terminal["type"] == "session_error"
        assert terminal["stage"] == "preflight-accept"
    finally:
        os.close(input_fd)
        os.close(output_write)
        os.close(output_read)


def test_open_ended_worker_returns_only_exact_binary_batch_evidence(monkeypatch):
    config = _config()
    session, launch = "6" * 32, _digest("a")
    _bind_init(monkeypatch, config, launch)
    requests = [
        batch_request(
            session_id=session,
            launch_digest=launch,
            request_id=character * 32,
            nonce=nonce * 32,
            batch_index=index,
            prompts=(f"prompt-{index}",),
            max_new_tokens=2,
            top_logprobs_num=1,
            temperature=0.0,
        )
        for index, (character, nonce) in enumerate((("7", "8"), ("9", "b")))
    ]
    payload = (
        _init_frame(config, session=session, launch=launch)
        + _accept_frame(config, session=session, launch=launch)
        + b"".join(
        frame_message(request, max_bytes=MAX_BATCH_REQUEST_BYTES)
        for request in requests
        )
    )
    input_fd, output_read, output_write = _session_fds(payload)
    completions: list[int] = []

    class Engine:
        def generate(self, **kwargs):
            assert kwargs["sampling_params"]["ignore_eos"] is True
            return {
                "output_ids": [11, 12],
                "meta_info": {
                    "output_top_logprobs": [
                        [[-0.25, 11, "forbidden-text"]],
                        [[-0.5, 12, "forbidden-text"]],
                    ]
                },
                "text": "must-not-cross-wire",
            }

    monkeypatch.setattr(
        worker,
        "_validate_live_preflight",
        lambda _config, *, launch_digest: (
            _facts(config, launch_digest), SimpleNamespace(root="tree")
        ),
    )

    @contextlib.contextmanager
    def engine_session(_config, _tree):
        yield worker._EngineHandle(
            Engine(), lambda: completions.append(len(completions))
        )

    monkeypatch.setattr(worker, "_engine_session", engine_session)
    try:
        assert worker.run_session(input_fd=input_fd, output_fd=output_write) == 1
        os.close(output_write)
        preflight = _read_frame(output_read)
        ready = _read_frame(output_read)
        evidence_frames = [_read_frame(output_read), _read_frame(output_read)]
        terminal = _read_frame(output_read)
        assert parse_frame_bytes(preflight, max_bytes=MAX_CONTROL_BYTES)["type"] == "preflight"
        assert parse_frame_bytes(ready, max_bytes=MAX_CONTROL_BYTES)["type"] == "ready"
        for raw, request_row in zip(evidence_frames, requests, strict=True):
            assert raw[:4] == EVIDENCE_MAGIC
            request = validate_batch_request(request_row)
            evidence = parse_evidence_frame_bytes(raw, request=request)
            assert evidence.prompts[0].output_ids == (11, 12)
            assert b"forbidden-text" not in raw and b"must-not-cross-wire" not in raw
        error = parse_frame_bytes(terminal, max_bytes=MAX_CONTROL_BYTES)
        assert error["type"] == "session_error"
        assert error["request_id"] is None
        assert completions == [0, 1]
        assert os.read(output_read, 1) == b""
    finally:
        os.close(input_fd)
        try:
            os.close(output_write)
        except OSError:
            pass
        os.close(output_read)


def test_protocol_fd_is_cloexec_and_engine_stdout_is_silenced(tmp_path):
    script = tmp_path / "fd_probe.py"
    script.write_text(
        "import os\n"
        "from optima.eval.oci_session_worker import _reserve_protocol_fd\n"
        "fd = _reserve_protocol_fd()\n"
        "assert not os.get_inheritable(fd)\n"
        "print('engine-noise', flush=True)\n"
        "os.write(fd, b'protocol-only')\n"
    )
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == b"protocol-only"
    assert result.stderr == b""


def test_descendant_pythonpath_contains_only_installed_site_bootstrap(
    monkeypatch,
):
    monkeypatch.setenv("PYTHONPATH", "/host/repository:/miner/tree")
    monkeypatch.setattr(worker, "_path_is_read_only", lambda _path: True)
    worker._prepare_descendant_bootstrap()
    site = Path(os.environ["PYTHONPATH"])
    assert site == Path(worker.__file__).resolve().parent / "oci_site"
    assert (site / "sitecustomize.py").is_file()
    assert "/host/repository" not in os.environ["PYTHONPATH"]
    assert "/miner/tree" not in os.environ["PYTHONPATH"]
