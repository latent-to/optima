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

from optima.eval import engine_worker as engine_policy
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
from optima.eval.reference_protocol import (
    EVIDENCE_MAGIC as REFERENCE_EVIDENCE_MAGIC,
    ReferencePromptInput,
    ReferenceRequest,
    ReferenceRoleInput,
    decode_reference_evidence,
    encode_reference_request,
)
from optima.seams import seam_binding_environment


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


def _reference_request(*, session: str, launch: str, index: int) -> ReferenceRequest:
    role = ReferenceRoleInput((7, 8), ((7, 9), (8, 10)))
    return ReferenceRequest(
        session,
        launch,
        _digest("3"),
        f"{index + 7:x}" * 32,
        f"{index + 10:x}" * 32,
        index,
        2,
        2,
        (ReferencePromptInput(_digest("6"), f"prompt-{index}", (role, role, role)),),
    )


class _ReferenceTokenizer:
    vocab_size = 128

    def __len__(self):
        return self.vocab_size

    def encode(self, _prompt):
        return [1, 2]


class _ReferenceEngine:
    tokenizer_manager = SimpleNamespace(tokenizer=_ReferenceTokenizer())

    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        rows = []
        for full_ids, requested in zip(
            kwargs["input_ids"], kwargs["token_ids_logprob"], strict=True
        ):
            response = full_ids[-2:]
            targets = [[None, 2, None]]
            top_one = [None]
            targeted = [None]
            for token in response:
                targets.append([-token / 100.0, token, None])
                top_one.append([[-requested[0] / 100.0, requested[0], None]])
                targeted.append(
                    [[-support / 100.0, support, None] for support in requested]
                )
            rows.append({
                "meta_info": {
                    "input_token_logprobs": targets,
                    "input_top_logprobs": top_one,
                    "input_token_ids_logprobs": targeted,
                }
            })
        return rows


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


def test_session_worker_transports_closed_seam_bindings_and_clears_reference(
    monkeypatch, tmp_path
):
    config = replace(_config(), seam_bindings=("collective",))
    gate_names = tuple(seam_binding_environment(()))
    observed = []
    bootstrap_gates = []
    manifest_gates = []

    class Engine:
        def __init__(self, **_kwargs):
            observed.append(
                {
                    "active": os.environ.get("OPTIMA_ACTIVE"),
                    "gates": {name: os.environ.get(name) for name in gate_names},
                    "plugins": os.environ.get("SGLANG_PLUGINS"),
                }
            )

        def shutdown(self):
            return None

    monkeypatch.setitem(sys.modules, "sglang", SimpleNamespace(Engine=Engine))
    monkeypatch.setenv("OPTIMA_EXTERNAL_NO_EGRESS", "1")
    monkeypatch.setenv("OPTIMA_ENGINE_WORKER", "1")
    for name in gate_names:
        monkeypatch.setenv(name, "1")
    for name in (
        "_loopback_is_up",
        "_network_namespace_is_loopback_only",
        "_egress_is_blocked",
        "_process_sandbox_is_hardened",
    ):
        monkeypatch.setattr(engine_policy, name, lambda: True)
    monkeypatch.setattr(
        worker,
        "_prepare_descendant_bootstrap",
        lambda: bootstrap_gates.append(
            {name: os.environ.get(name) for name in gate_names}
        ),
    )

    from optima import manifest as manifest_module
    from optima import receipts, seam

    monkeypatch.setattr(seam, "mark_driver", lambda: None)
    monkeypatch.setattr(
        receipts,
        "require",
        lambda *_args, **_kwargs: [
            {"pid": 1, "slots": ["collective.all_reduce"]}
        ],
    )

    def load_manifest(_root):
        manifest_gates.append(
            {name: os.environ.get(name) for name in gate_names}
        )
        return SimpleNamespace(
            ops=(SimpleNamespace(setup=None),), dep_patches=()
        )

    monkeypatch.setattr(manifest_module, "load_manifest", load_manifest)

    baseline = SimpleNamespace(root=tmp_path / "baseline", runtime_manifest=None)
    candidate = SimpleNamespace(
        root=tmp_path / "candidate", runtime_manifest="manifest.toml"
    )
    monkeypatch.delenv("OPTIMA_SESSION_PROTOCOL", raising=False)
    with worker._engine_session(config, baseline):
        pass
    with worker._engine_session(config, candidate):
        pass
    monkeypatch.setenv("OPTIMA_SESSION_PROTOCOL", "reference")
    with pytest.raises(worker.SessionWorkerError, match="reference.*seam bindings"):
        with worker._engine_session(config, baseline):
            pass
    with worker._engine_session(replace(config, seam_bindings=()), baseline):
        pass

    selected = seam_binding_environment(("collective",))
    cleared = seam_binding_environment(())
    assert observed == [
        {"active": "0", "gates": selected, "plugins": "optima"},
        {"active": "1", "gates": selected, "plugins": "optima"},
        {"active": "0", "gates": cleared, "plugins": ""},
    ]
    assert bootstrap_gates == [selected, selected]
    assert manifest_gates == [selected]


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
        yield SimpleNamespace(engine=SimpleNamespace(), require_completion=lambda: None)

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
        yield SimpleNamespace(
            engine=Engine(),
            require_completion=lambda: completions.append(len(completions)),
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


def test_reference_worker_serves_ordered_requests_from_one_pristine_engine(monkeypatch):
    config = _config()
    session, launch = "6" * 32, _digest("a")
    requests = tuple(
        _reference_request(session=session, launch=launch, index=index)
        for index in range(2)
    )
    _bind_init(monkeypatch, config, launch)
    monkeypatch.setenv("OPTIMA_SESSION_PROTOCOL", "reference")
    payload = (
        _init_frame(config, session=session, launch=launch)
        + _accept_frame(config, session=session, launch=launch)
        + b"".join(encode_reference_request(request) for request in requests)
    )
    input_fd, output_read, output_write = _session_fds(payload)
    engine = _ReferenceEngine()
    monkeypatch.setattr(
        worker,
        "_validate_live_preflight",
        lambda _config, *, launch_digest: (
            _facts(config, launch_digest),
            SimpleNamespace(root="tree", runtime_manifest=None),
        ),
    )

    @contextlib.contextmanager
    def engine_session(_config, _tree):
        yield SimpleNamespace(engine=engine, require_completion=lambda: None)

    monkeypatch.setattr(worker, "_engine_session", engine_session)
    try:
        assert worker.run_session(input_fd=input_fd, output_fd=output_write) == 1
        os.close(output_write)
        assert parse_frame_bytes(
            _read_frame(output_read), max_bytes=MAX_CONTROL_BYTES
        )["type"] == "preflight"
        assert parse_frame_bytes(
            _read_frame(output_read), max_bytes=MAX_CONTROL_BYTES
        )["type"] == "ready"
        for request in requests:
            frame = _read_frame(output_read)
            assert frame[:4] == REFERENCE_EVIDENCE_MAGIC
            evidence = decode_reference_evidence(frame, request)
            assert evidence.request_index == request.request_index
            assert evidence.prompts[0].prompt_token_count == 2
            assert evidence.prompts[0].roles[0].tokens[0].target_logprob == pytest.approx(-0.07)
        terminal = parse_frame_bytes(
            _read_frame(output_read), max_bytes=MAX_CONTROL_BYTES
        )
        assert terminal["type"] == "session_error"
        assert terminal["stage"] == "reference"
        assert len(engine.calls) == 6
        assert all(call["top_logprobs_num"] == 1 for call in engine.calls)
        assert os.read(output_read, 1) == b""
    finally:
        os.close(input_fd)
        try:
            os.close(output_write)
        except OSError:
            pass
        os.close(output_read)


def test_reference_worker_rejects_any_contribution_manifest_before_engine(monkeypatch):
    config = _config()
    session, launch = "6" * 32, _digest("a")
    _bind_init(monkeypatch, config, launch)
    monkeypatch.setenv("OPTIMA_SESSION_PROTOCOL", "reference")
    input_fd, output_read, output_write = _session_fds(
        _init_frame(config, session=session, launch=launch)
        + _accept_frame(config, session=session, launch=launch)
    )
    entered = []
    monkeypatch.setattr(
        worker,
        "_validate_live_preflight",
        lambda _config, *, launch_digest: (
            _facts(config, launch_digest),
            SimpleNamespace(root="tree", runtime_manifest="manifest.toml"),
        ),
    )

    @contextlib.contextmanager
    def forbidden_engine(*_args, **_kwargs):
        entered.append(True)
        yield

    monkeypatch.setattr(worker, "_engine_session", forbidden_engine)
    try:
        assert worker.run_session(input_fd=input_fd, output_fd=output_write) == 1
        os.close(output_write)
        assert parse_frame_bytes(
            _read_frame(output_read), max_bytes=MAX_CONTROL_BYTES
        )["type"] == "preflight"
        terminal = parse_frame_bytes(
            _read_frame(output_read), max_bytes=MAX_CONTROL_BYTES
        )
        assert terminal["type"] == "session_error"
        assert terminal["stage"] == "engine"
        assert entered == []
    finally:
        os.close(input_fd)
        try:
            os.close(output_write)
        except OSError:
            pass
        os.close(output_read)


def test_protocol_fd_is_cloexec_stdout_is_silenced_and_stderr_is_separate(tmp_path):
    script = tmp_path / "fd_probe.py"
    script.write_text(
        "import os\n"
        "from optima.eval.oci_session_worker import _reserve_protocol_fd\n"
        "fd = _reserve_protocol_fd()\n"
        "assert not os.get_inheritable(fd)\n"
        "print('engine-noise', flush=True)\n"
        "print('scheduler-traceback', file=__import__('sys').stderr, flush=True)\n"
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
    assert result.stderr == b"scheduler-traceback\n"


def test_worker_fallback_stderr_diagnostic_has_a_hard_byte_cap(tmp_path):
    script = tmp_path / "stderr_probe.py"
    script.write_text(
        "from optima.eval.oci_session_worker import _write_stderr_diagnostic\n"
        "_write_stderr_diagnostic('batch', RuntimeError('x' * 100000))\n"
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
    assert result.stdout == b""
    assert result.stderr.startswith(b"[optima-session-worker]")
    assert 0 < len(result.stderr) <= worker.WORKER_STDERR_DIAGNOSTIC_MAX_BYTES


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
