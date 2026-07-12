from __future__ import annotations

import hashlib
import inspect
import os
import struct
import time
from dataclasses import replace

import pytest

import optima.eval.oci_outer_session as outer
from optima.eval.oci_outer_session import (
    AttachedSessionTransport,
    OuterSessionInfrastructureError,
    OuterSessionProcessError,
    OuterSessionProtocolError,
    OuterSessionTimeoutError,
    OuterSessionWorkerError,
    SessionExecutionPlan,
    run_outer_session,
)
from optima.eval.oci_session_protocol import (
    CONTROL_MAGIC,
    EVIDENCE_MAGIC,
    MAX_CONTROL_BYTES,
    BatchEvidence,
    BatchRequest,
    EngineSessionConfig,
    PromptEvidence,
    RuntimePreflightFacts,
    batch_request,
    error_message,
    evidence_frame,
    frame_message,
    parse_frame_bytes,
    preflight_message,
    ready_message,
    validate_batch_request,
    validate_init,
    validate_preflight_accept,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


LAUNCH = _digest("launch")


def _config() -> EngineSessionConfig:
    return EngineSessionConfig(
        model_path="/optima/input/model",
        dtype="bfloat16",
        deterministic=False,
        attention_backend="flashinfer",
        disable_cuda_graph=False,
        mem_fraction_static=0.82,
        log_level="error",
        max_running_requests=64,
        tp_size=8,
        moe_runner_backend="flashinfer_trtllm",
        disable_custom_all_reduce=False,
        engine_kwargs={"page_size": 64},
    )


def _facts(config: EngineSessionConfig | None = None) -> RuntimePreflightFacts:
    cfg = config or _config()
    return RuntimePreflightFacts(
        launch_digest=LAUNCH,
        runtime_digest=_digest("runtime"),
        stack_digest=_digest("stack"),
        tree_digest=_digest("tree"),
        engine_config_digest=cfg.digest,
        worker_distribution_digest=_digest("worker"),
        model_revision_digest=_digest("revision"),
        model_manifest_digest=_digest("manifest"),
        model_content_digest=_digest("content"),
        sglang_version="0.0.0.dev1+g56e290315",
        gpu_architectures=("sm120",) * 8,
        topology_digest=_digest("topology"),
        loopback_only=True,
        read_only_inputs=True,
        private_writable_cache=True,
    )


def _plan(**changes: object) -> SessionExecutionPlan:
    config = _config()
    values: dict[str, object] = {
        "launch_digest": LAUNCH,
        "expected_engine_config_digest": config.digest,
        "engine_config": config,
        "expected_preflight": _facts(config),
        "prompt_batches": (("a", "b"), ("c", "d"), ("e", "f")),
        "warmup_count": 2,
        "conditioning_count": 1,
        "max_new_tokens": 2,
        "top_logprobs_num": 2,
        "temperature": 0.0,
    }
    values.update(changes)
    return SessionExecutionPlan(**values)  # type: ignore[arg-type]


def _batch_evidence(request: BatchRequest) -> BatchEvidence:
    prompts = []
    for prompt_index in range(len(request.prompts)):
        ids = tuple(prompt_index * 10 + index for index in range(request.max_new_tokens))
        positions = tuple(
            ((-0.7, 100 + index), (-1.0, 200 + index))
            for index in range(request.max_new_tokens)
        )
        prompts.append(PromptEvidence(ids, positions))
    return BatchEvidence(tuple(prompts))


class _Clock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _FakeTransport:
    def __init__(
        self,
        clock: _Clock,
        facts: RuntimePreflightFacts,
        *,
        init_write_s: float = 0.1,
        control_read_s: float = 0.1,
        batch_write_s: float = 0.5,
        batch_read_s: float = 1.5,
        pending_calls: set[int] | None = None,
        preflight_override: dict | None = None,
        ready_override: dict | None = None,
    ) -> None:
        self.clock = clock
        self.facts = facts
        self.init_write_s = init_write_s
        self.control_read_s = control_read_s
        self.batch_write_s = batch_write_s
        self.batch_read_s = batch_read_s
        self.pending_calls = pending_calls or set()
        self.preflight_override = preflight_override
        self.ready_override = ready_override
        self.started = False
        self.finalized = False
        self.aborted = False
        self.pending_count = 0
        self.session_id = ""
        self.launch_digest = ""
        self.control_reads = 0
        self.accepted = False
        self.requests: list[BatchRequest] = []
        self.messages: list[dict] = []
        self.deadlines: list[tuple[str, float]] = []

    def _advance(self, seconds: float, deadline: float, stage: str) -> None:
        self.deadlines.append((stage, deadline))
        self.clock.advance(seconds)
        if self.clock() > deadline:
            raise OuterSessionTimeoutError(f"{stage} timed out")

    def start(self) -> None:
        self.started = True

    def has_pending_output(self) -> bool:
        self.pending_count += 1
        return self.pending_count in self.pending_calls

    def write_frame(self, frame: bytes, *, deadline: float) -> None:
        message = parse_frame_bytes(frame, max_bytes=outer.MAX_BATCH_REQUEST_BYTES)
        self.messages.append(message)
        if message.get("type") == "init":
            self.session_id, self.launch_digest, _ = validate_init(message)
            self._advance(self.init_write_s, deadline, "init-write")
        elif message.get("type") == "preflight_accept":
            validate_preflight_accept(
                message,
                session_id=self.session_id,
                launch_digest=self.launch_digest,
                expected_facts_digest=self.facts.digest,
            )
            self.accepted = True
            self._advance(self.init_write_s, deadline, "accept-write")
        else:
            request = validate_batch_request(message)
            assert request.batch_index == len(self.requests)
            self.requests.append(request)
            self._advance(self.batch_write_s, deadline, "batch-write")

    def read_control(self, *, max_bytes: int, deadline: float) -> dict:
        self._advance(self.control_read_s, deadline, "control-read")
        self.control_reads += 1
        if self.control_reads == 1:
            return self.preflight_override or preflight_message(
                session_id=self.session_id,
                launch_digest=self.launch_digest,
                facts=self.facts,
            )
        if not self.accepted:
            raise AssertionError("ready was requested before host preflight acceptance")
        return self.ready_override or ready_message(
            session_id=self.session_id, launch_digest=self.launch_digest
        )

    def read_evidence(self, request: BatchRequest, *, deadline: float) -> BatchEvidence:
        assert request == self.requests[-1]
        self._advance(self.batch_read_s, deadline, "batch-read")
        return _batch_evidence(request)

    def finalize(self) -> None:
        self.finalized = True
        self.clock.advance(0.25)

    def abort(self) -> None:
        self.aborted = True


def test_plan_validates_every_frame_before_start_and_contains_no_execution_policy() -> None:
    plan = _plan()
    assert plan.engine_config.digest == plan.expected_engine_config_digest
    for changes, match in (
        ({"warmup_count": 0}, "warmup"),
        ({"warmup_count": 3}, "warmup"),
        ({"conditioning_count": 0}, "conditioning"),
        ({"prompt_batches": ("not-a-batch",)}, "each prompt batch"),
        ({"prompt_batches": (("ok",), (), ("timed",))}, "controller batch"),
        ({"expected_engine_config_digest": _digest("wrong")}, "digest"),
        ({"expected_preflight": replace(_facts(), launch_digest=_digest("wrong"))}, "preflight"),
    ):
        with pytest.raises(OuterSessionInfrastructureError, match=match):
            _plan(**changes)

    assert not hasattr(plan, "mode")
    assert not hasattr(plan, "role")
    assert not hasattr(plan, "score")


def test_happy_path_accepts_preflight_before_ready_and_returns_raw_host_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = iter(f"{index:032x}" for index in range(1, 20))
    monkeypatch.setattr(outer.secrets, "token_hex", lambda _count: next(ids))
    clock = _Clock()
    plan = _plan()
    transport = _FakeTransport(clock, plan.expected_preflight)
    callbacks: list[tuple[str, int, float]] = []

    def boundary(event: str, index: int, deadline: float) -> None:
        callbacks.append((event, index, deadline))
        clock.advance(3.0)

    result = run_outer_session(
        plan,
        transport=transport,
        deadline=1000.0,
        init_timeout_s=30.0,
        batch_timeout_s=30.0,
        clock=clock,
        boundary_callback=boundary,
    )

    assert transport.finalized and not transport.aborted
    assert [message["type"] for message in transport.messages] == [
        "init",
        "preflight_accept",
        "batch_request",
        "batch_request",
        "batch_request",
    ]
    init = transport.messages[0]
    assert set(init) == {
        "engine_config", "engine_config_digest", "launch_digest", "schema",
        "session_id", "type",
    }
    assert not ({"prompts", "seed", "entropy", "threshold", "mode", "role"} & set(init))
    for request in transport.messages[2:]:
        assert not ({"warmup", "timed", "mode", "role", "score"} & set(request))
    assert callbacks == [
        ("before_final_warmup", 1, 1000.0),
        ("after_final_warmup", 1, 1000.0),
        ("before_first_timed", 2, 1000.0),
    ]
    assert len({result.session_id, *(row.request_id for row in result.batches),
                *(row.nonce for row in result.batches)}) == 7
    assert all(row.elapsed_seconds == pytest.approx(2.0) for row in result.batches)
    assert result.conditioning_started_at == result.batches[0].response_completed_at
    assert result.first_timed_completed_at == result.batches[2].response_completed_at
    assert result.conditioning_token_numerator == 8
    assert result.conditioning_interval_seconds > (
        result.batches[1].elapsed_seconds + result.batches[2].elapsed_seconds
    )
    assert result.preflight == plan.expected_preflight


def test_preflight_rejection_never_sends_accept_or_requests() -> None:
    clock = _Clock()
    plan = _plan()
    wrong = preflight_message(
        session_id="1" * 32,
        launch_digest=LAUNCH,
        facts=replace(plan.expected_preflight, tree_digest=_digest("wrong-tree")),
    )
    transport = _FakeTransport(clock, plan.expected_preflight, preflight_override=wrong)
    with pytest.raises(OuterSessionInfrastructureError, match="preflight"):
        run_outer_session(
            plan,
            transport=transport,
            deadline=1000.0,
            init_timeout_s=30.0,
            batch_timeout_s=30.0,
            clock=clock,
        )
    assert [message["type"] for message in transport.messages] == ["init"]
    assert transport.aborted and not transport.finalized


@pytest.mark.parametrize(
    ("transport_kwargs", "init_cap", "batch_cap", "expected_stage"),
    [
        ({"init_write_s": 0.4, "control_read_s": 0.4}, 0.5, 30.0, "control-read"),
        ({"batch_write_s": 0.4, "batch_read_s": 0.4}, 30.0, 0.5, "batch-read"),
    ],
)
def test_init_and_batch_phase_deadlines_are_independent(
    transport_kwargs: dict[str, float],
    init_cap: float,
    batch_cap: float,
    expected_stage: str,
) -> None:
    clock = _Clock()
    plan = _plan()
    transport = _FakeTransport(clock, plan.expected_preflight, **transport_kwargs)
    with pytest.raises(OuterSessionTimeoutError, match=expected_stage):
        run_outer_session(
            plan,
            transport=transport,
            deadline=1000.0,
            init_timeout_s=init_cap,
            batch_timeout_s=batch_cap,
            clock=clock,
        )
    assert transport.aborted and not transport.finalized


def test_caller_absolute_deadline_is_not_rebased_by_session() -> None:
    clock = _Clock()
    plan = _plan()
    transport = _FakeTransport(
        clock,
        plan.expected_preflight,
        init_write_s=0.3,
        control_read_s=0.3,
    )
    with pytest.raises(OuterSessionTimeoutError):
        run_outer_session(
            plan,
            transport=transport,
            deadline=100.7,
            init_timeout_s=30.0,
            batch_timeout_s=30.0,
            clock=clock,
        )
    assert transport.aborted


@pytest.mark.parametrize(("pending_call", "match"), [(1, "before init"), (2, "first request"), (4, "trailing")])
def test_early_and_trailing_output_is_terminal(pending_call: int, match: str) -> None:
    clock = _Clock()
    plan = _plan()
    transport = _FakeTransport(
        clock, plan.expected_preflight, pending_calls={pending_call}
    )
    with pytest.raises(OuterSessionProtocolError, match=match):
        run_outer_session(
            plan,
            transport=transport,
            deadline=1000.0,
            init_timeout_s=30.0,
            batch_timeout_s=30.0,
            clock=clock,
        )
    assert transport.aborted


def test_duplicate_internal_binding_is_infrastructure_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(outer.secrets, "token_hex", lambda _count: "1" * 32)
    clock = _Clock()
    plan = _plan()
    transport = _FakeTransport(clock, plan.expected_preflight)
    with pytest.raises(OuterSessionInfrastructureError, match="RNG repeated"):
        run_outer_session(
            plan,
            transport=transport,
            deadline=1000.0,
            init_timeout_s=30.0,
            batch_timeout_s=30.0,
            clock=clock,
        )
    assert transport.aborted


class _PipeClient:
    def __init__(self) -> None:
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        self.stdin = os.fdopen(request_write, "wb", buffering=0)
        self.stdout = os.fdopen(response_read, "rb", buffering=0)
        self.request_read = request_read
        self.response_write = response_write
        self.closed = False
        self.finalized = False
        self.aborted = False

    def finalize(self) -> None:
        self.finalized = True
        self.closed = True

    def abort(self) -> None:
        self.aborted = True
        self.closed = True

    def close_fds(self) -> None:
        for stream in (self.stdin, self.stdout):
            if not stream.closed:
                stream.close()
        for fd in (self.request_read, self.response_write):
            try:
                os.close(fd)
            except OSError:
                pass


class _PipeManager:
    def __init__(self, client: _PipeClient) -> None:
        self.client = client
        self.calls = 0

    def spawn_attached(self, _lease, _argv):
        self.calls += 1
        return self.client


def _attached() -> tuple[AttachedSessionTransport, _PipeClient]:
    client = _PipeClient()
    transport = AttachedSessionTransport(
        _PipeManager(client), object(), ("/usr/bin/docker", "run")  # type: ignore[arg-type]
    )
    transport.start()
    return transport, client


def _request(index: int = 0, *, request_id: str = "2" * 32) -> BatchRequest:
    return validate_batch_request(batch_request(
        session_id="1" * 32,
        launch_digest=LAUNCH,
        request_id=request_id,
        nonce=("3" if index == 0 else "5") * 32,
        batch_index=index,
        prompts=("a",),
        max_new_tokens=2,
        top_logprobs_num=2,
        temperature=0.0,
    ))


def test_attached_transport_is_nonblocking_and_manager_owns_both_teardown_paths() -> None:
    source = inspect.getsource(AttachedSessionTransport)
    assert ".poll(" not in source
    assert ".wait(" not in source
    assert "force_remove" not in source
    assert "os.kill" not in source

    transport, client = _attached()
    try:
        assert not os.get_blocking(client.stdin.fileno())
        assert not os.get_blocking(client.stdout.fileno())
        message = ready_message(session_id="1" * 32, launch_digest=LAUNCH)
        frame = frame_message(message, max_bytes=MAX_CONTROL_BYTES)
        transport.write_frame(frame, deadline=time.monotonic() + 1)
        assert os.read(client.request_read, len(frame)) == frame
        os.write(client.response_write, frame)
        assert transport.read_control(
            max_bytes=MAX_CONTROL_BYTES, deadline=time.monotonic() + 1
        ) == message
        transport.finalize()
        assert client.finalized and not client.aborted
    finally:
        client.close_fds()

    transport, client = _attached()
    try:
        transport.abort()
        assert client.aborted and not client.finalized
    finally:
        client.close_fds()


def test_attached_transport_rejects_partial_wrong_magic_oversized_and_timeout() -> None:
    cases = (
        (b"OES1", OuterSessionProcessError, "complete"),
        (b"NOPE" + struct.pack(">I", 0), OuterSessionProtocolError, "magic"),
        (
            CONTROL_MAGIC + struct.pack(">I", MAX_CONTROL_BYTES + 1),
            OuterSessionProtocolError,
            "oversized",
        ),
    )
    for raw, error, match in cases:
        transport, client = _attached()
        try:
            os.write(client.response_write, raw)
            if len(raw) < 8:
                os.close(client.response_write)
                client.response_write = -1
            with pytest.raises(error, match=match):
                transport.read_control(
                    max_bytes=MAX_CONTROL_BYTES, deadline=time.monotonic() + 1
                )
        finally:
            transport.abort()
            client.close_fds()

    transport, client = _attached()
    try:
        with pytest.raises(OuterSessionTimeoutError, match="timed out"):
            transport.read_control(
                max_bytes=MAX_CONTROL_BYTES, deadline=time.monotonic() + 0.01
            )
    finally:
        transport.abort()
        client.close_fds()


def test_attached_transport_rejects_replay_error_and_trailing_bytes() -> None:
    current = _request(1, request_id="4" * 32)
    stale = _request(0)
    transport, client = _attached()
    try:
        os.write(client.response_write, evidence_frame(_batch_evidence(stale), request=stale))
        with pytest.raises(OuterSessionProtocolError, match="binding"):
            transport.read_evidence(current, deadline=time.monotonic() + 1)
    finally:
        transport.abort()
        client.close_fds()

    transport, client = _attached()
    try:
        error = error_message(
            session_id=current.session_id,
            launch_digest=current.launch_digest,
            stage="batch-1",
            error=RuntimeError("failed"),
            request=current,
        )
        os.write(client.response_write, frame_message(error, max_bytes=MAX_CONTROL_BYTES))
        with pytest.raises(OuterSessionWorkerError, match="failed"):
            transport.read_evidence(current, deadline=time.monotonic() + 1)
    finally:
        transport.abort()
        client.close_fds()

    transport, client = _attached()
    try:
        frame = evidence_frame(_batch_evidence(current), request=current)
        os.write(client.response_write, frame + b"x")
        assert transport.read_evidence(current, deadline=time.monotonic() + 1)
        assert transport.has_pending_output()
    finally:
        transport.abort()
        client.close_fds()
