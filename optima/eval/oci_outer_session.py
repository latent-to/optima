"""Trusted host transport and raw timing evidence for one isolated engine.

This module owns framed byte I/O and the host clock.  It never imports an
inference runtime, interprets quality, assigns an arm role, or accepts worker
timing.  OCI policy and resource construction belong to :mod:`oci_backend`;
process creation and cleanup remain exclusively manager-owned.
"""

from __future__ import annotations

import math
import os
import secrets
import select
import struct
import time
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from optima.discovery_overlay import DiscoveryActivationReceipt
from optima.eval.oci_process import (
    OCIAttachedClient,
    OCIAttachedDiagnostic,
    OCILease,
    OCIProcessError,
    OCIProcessManager,
)
from optima.eval.oci_session_protocol import (
    CONTROL_MAGIC,
    EVIDENCE_MAGIC,
    FRAME_HEADER_BYTES,
    MAX_BATCH_REQUEST_BYTES,
    MAX_BATCH_RESPONSE_BYTES,
    MAX_CONTROL_BYTES,
    MAX_INIT_BYTES,
    BatchEvidence,
    BatchRequest,
    EngineSessionConfig,
    RuntimePreflightFacts,
    SessionProtocolError,
    batch_request,
    decode_evidence_payload,
    decode_message,
    expected_evidence_payload_bytes,
    frame_message,
    make_init,
    parse_error_message,
    preflight_accept_message,
    validate_batch_request,
    validate_preflight,
    validate_ready,
)
from optima.stack_identity import require_sha256_hex


class OuterSessionError(RuntimeError):
    """Base error for host transport and raw session execution."""

    def __init__(
        self,
        message: str,
        diagnostic_provider: Callable[[], OCIAttachedDiagnostic] | None = None,
    ) -> None:
        super().__init__(message)
        self._message = message
        self._diagnostic_provider = diagnostic_provider

    @property
    def message(self) -> str:
        return self._message

    def attach_diagnostic(
        self, provider: Callable[[], OCIAttachedDiagnostic] | None
    ) -> None:
        """Attach host-only failure evidence without copying candidate bytes."""

        if self._diagnostic_provider is None and callable(provider):
            self._diagnostic_provider = provider

    @property
    def diagnostic(self) -> OCIAttachedDiagnostic | None:
        if self._diagnostic_provider is None:
            return None
        try:
            value = self._diagnostic_provider()
        except BaseException:
            return None
        return value if type(value) is OCIAttachedDiagnostic else None

    def __str__(self) -> str:
        diagnostic = self.diagnostic
        if diagnostic is None:
            return self._message
        return f"{self._message}; {diagnostic.summary}"


class OuterSessionInfrastructureError(OuterSessionError):
    """Trusted host, OCI lifecycle, or pre-entry runtime failure."""


class OuterSessionTimeoutError(OuterSessionInfrastructureError):
    """The one absolute session deadline expired."""


class OuterSessionProcessError(OuterSessionInfrastructureError):
    """The attached client closed a protocol pipe before completion."""


class OuterSessionProtocolError(OuterSessionError):
    """The worker emitted malformed, stale, early, or extra protocol bytes."""


class OuterSessionWorkerError(OuterSessionError):
    """The worker emitted one valid, bounded error control frame."""


class SessionTransport(Protocol):
    def start(self) -> None: ...
    def has_pending_output(self) -> bool: ...
    def write_frame(self, frame: bytes, *, deadline: float) -> None: ...
    def read_control(self, *, max_bytes: int, deadline: float) -> dict: ...
    def read_evidence(self, request: BatchRequest, *, deadline: float) -> BatchEvidence: ...
    def finalize(self) -> None: ...
    def abort(self) -> None: ...


class AttachedSessionTransport:
    """Nonblocking pipes around one manager-owned attached OCI client."""

    def __init__(
        self,
        manager: OCIProcessManager,
        lease: OCILease,
        argv: Sequence[str],
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(argv, (str, bytes)) or not argv:
            raise OuterSessionInfrastructureError("attached session argv is invalid")
        self.manager = manager
        self.lease = lease
        self.argv = tuple(argv)
        self.clock = clock
        self.client: OCIAttachedClient | None = None
        self._stdin_fd = -1
        self._stdout_fd = -1

    def start(self) -> None:
        if self.client is not None:
            raise OuterSessionInfrastructureError("attached session already started")
        try:
            client = self.manager.spawn_attached(self.lease, self.argv)
            self.client = client
            self._stdin_fd = client.stdin.fileno()
            self._stdout_fd = client.stdout.fileno()
            os.set_blocking(self._stdin_fd, False)
            os.set_blocking(self._stdout_fd, False)
        except (OSError, OCIProcessError) as exc:
            try:
                self.abort()
            except OuterSessionError as cleanup:
                raise OuterSessionInfrastructureError(
                    f"attached session start cleanup failed: {cleanup}"
                ) from exc
            raise OuterSessionInfrastructureError(
                f"could not start attached OCI session: {exc}"
            ) from None

    def _require_client(self) -> OCIAttachedClient:
        if self.client is None or self.client.closed:
            raise OuterSessionInfrastructureError("attached session is not live")
        return self.client

    def _process_error(self, message: str) -> OuterSessionProcessError:
        client = self.client
        provider = getattr(client, "stderr_diagnostic", None)
        return OuterSessionProcessError(
            message,
            provider if callable(provider) else None,
        )

    def stderr_diagnostic(self) -> OCIAttachedDiagnostic:
        client = self.client
        if client is None:
            raise OuterSessionInfrastructureError(
                "attached session has no stderr diagnostic"
            )
        return client.stderr_diagnostic()

    def _diagnostic_provider(
        self,
    ) -> Callable[[], OCIAttachedDiagnostic] | None:
        return self.stderr_diagnostic if self.client is not None else None

    def _diagnostic_error(
        self, error_type: type[OuterSessionError], message: str
    ) -> OuterSessionError:
        return error_type(message, self._diagnostic_provider())

    def _remaining(self, deadline: float) -> float:
        try:
            remaining = float(deadline) - float(self.clock())
        except Exception as exc:
            raise OuterSessionInfrastructureError(f"host clock failed: {exc}") from None
        if not math.isfinite(remaining) or remaining <= 0:
            raise OuterSessionTimeoutError("attached session deadline expired")
        return remaining

    def has_pending_output(self) -> bool:
        self._require_client()
        try:
            readable, _, _ = select.select([self._stdout_fd], [], [], 0)
        except OSError as exc:
            raise OuterSessionInfrastructureError(f"cannot inspect session output: {exc}") from None
        return bool(readable)

    def write_frame(self, frame: bytes, *, deadline: float) -> None:
        self._require_client()
        if not isinstance(frame, bytes) or not frame:
            raise OuterSessionInfrastructureError("session request frame is invalid")
        view = memoryview(frame)
        offset = 0
        while offset < len(view):
            try:
                _, writable, _ = select.select(
                    [], [self._stdin_fd], [], self._remaining(deadline)
                )
                if not writable:
                    raise OuterSessionTimeoutError("session request write timed out")
                count = os.write(self._stdin_fd, view[offset:])
            except (BlockingIOError, InterruptedError):
                continue
            except BrokenPipeError:
                raise self._process_error("session closed its request pipe") from None
            except OSError as exc:
                raise self._process_error(
                    f"session request write failed: {exc}"
                ) from None
            if count <= 0:
                raise self._process_error("session request write made no progress")
            offset += count

    def _read_exact(self, size: int, *, deadline: float) -> bytes:
        self._require_client()
        remaining = size
        chunks: list[bytes] = []
        while remaining:
            try:
                readable, _, _ = select.select(
                    [self._stdout_fd], [], [], self._remaining(deadline)
                )
                if not readable:
                    raise OuterSessionTimeoutError("session response read timed out")
                chunk = os.read(self._stdout_fd, min(remaining, 1 << 20))
            except (BlockingIOError, InterruptedError):
                continue
            except OSError as exc:
                raise self._process_error(
                    f"session response read failed: {exc}"
                ) from None
            if not chunk:
                # Never poll/wait here: only the manager may reap the process group.
                raise self._process_error(
                    "session ended before a complete response"
                )
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _header(self, *, deadline: float) -> tuple[bytes, int]:
        header = self._read_exact(FRAME_HEADER_BYTES, deadline=deadline)
        return header[:4], struct.unpack(">I", header[4:])[0]

    def read_control(self, *, max_bytes: int, deadline: float) -> dict:
        magic, size = self._header(deadline=deadline)
        if magic != CONTROL_MAGIC:
            raise OuterSessionProtocolError("worker emitted wrong control-frame magic")
        if size > max_bytes:
            raise OuterSessionProtocolError("worker declared an oversized control frame")
        try:
            return decode_message(self._read_exact(size, deadline=deadline), max_bytes=max_bytes)
        except SessionProtocolError as exc:
            raise OuterSessionProtocolError(str(exc)) from None

    def read_evidence(self, request: BatchRequest, *, deadline: float) -> BatchEvidence:
        magic, size = self._header(deadline=deadline)
        if magic == CONTROL_MAGIC:
            if size > MAX_CONTROL_BYTES:
                raise OuterSessionProtocolError("worker declared an oversized error frame")
            try:
                message = decode_message(
                    self._read_exact(size, deadline=deadline), max_bytes=MAX_CONTROL_BYTES
                )
                detail = parse_error_message(
                    message,
                    session_id=request.session_id,
                    launch_digest=request.launch_digest,
                    request=request,
                )
            except SessionProtocolError as exc:
                raise OuterSessionProtocolError(str(exc)) from None
            if detail is not None:
                raise self._diagnostic_error(
                    OuterSessionWorkerError, ": ".join(detail)
                )
            raise OuterSessionProtocolError("worker emitted an early control frame")
        if magic != EVIDENCE_MAGIC:
            raise OuterSessionProtocolError("worker emitted wrong evidence-frame magic")
        exact = expected_evidence_payload_bytes(request)
        if size != exact or size > MAX_BATCH_RESPONSE_BYTES:
            raise OuterSessionProtocolError("worker evidence frame has the wrong exact size")
        try:
            payload = self._read_exact(size, deadline=deadline)
            return decode_evidence_payload(payload, request=request)
        except SessionProtocolError as exc:
            raise OuterSessionProtocolError(str(exc)) from None

    def finalize(self) -> None:
        if self.client is None:
            return
        try:
            self.client.finalize()
        except OCIProcessError as exc:
            raise self._diagnostic_error(
                OuterSessionInfrastructureError, f"session cleanup failed: {exc}"
            ) from None

    def abort(self) -> None:
        if self.client is None or self.client.closed:
            return
        try:
            self.client.abort()
        except OCIProcessError as exc:
            raise self._diagnostic_error(
                OuterSessionInfrastructureError, f"session cleanup failed: {exc}"
            ) from None


@dataclass(frozen=True)
class SessionExecutionPlan:
    """Host-only session inputs; only one prompt batch crosses at a time."""

    launch_digest: str
    expected_engine_config_digest: str
    engine_config: EngineSessionConfig
    expected_preflight: RuntimePreflightFacts
    prompt_batches: tuple[tuple[str, ...], ...]
    warmup_count: int
    conditioning_count: int
    max_new_tokens: int
    top_logprobs_num: int
    temperature: float
    expected_discovery_overlay_identity_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.engine_config, EngineSessionConfig):
            raise OuterSessionInfrastructureError("engine_config is not typed")
        if self.engine_config.digest != self.expected_engine_config_digest:
            raise OuterSessionInfrastructureError("engine config digest differs from plan")
        if self.expected_discovery_overlay_identity_digest is not None:
            try:
                identity = require_sha256_hex(
                    self.expected_discovery_overlay_identity_digest,
                    field="expected discovery overlay identity",
                )
            except ValueError as exc:
                raise OuterSessionInfrastructureError(str(exc)) from None
            if identity == "0" * 64:
                raise OuterSessionInfrastructureError(
                    "expected discovery overlay identity must not be all zero"
                )
        if (
            not isinstance(self.expected_preflight, RuntimePreflightFacts)
            or self.expected_preflight.launch_digest != self.launch_digest
            or self.expected_preflight.engine_config_digest
            != self.expected_engine_config_digest
        ):
            raise OuterSessionInfrastructureError("preflight facts differ from plan identity")
        if isinstance(self.prompt_batches, (str, bytes)) or not isinstance(
            self.prompt_batches, Sequence
        ):
            raise OuterSessionInfrastructureError("prompt_batches must be a sequence")
        if any(isinstance(batch, (str, bytes)) for batch in self.prompt_batches):
            raise OuterSessionInfrastructureError("each prompt batch must be a sequence")
        try:
            batches = tuple(tuple(batch) for batch in self.prompt_batches)
        except TypeError:
            raise OuterSessionInfrastructureError("each prompt batch must be a sequence") from None
        if not batches or type(self.warmup_count) is not int or not 1 <= self.warmup_count < len(batches):
            raise OuterSessionInfrastructureError("session requires warmup and timed batches")
        if type(self.conditioning_count) is not int or not 1 <= self.conditioning_count <= self.warmup_count:
            raise OuterSessionInfrastructureError("conditioning_count must be in 1..warmup_count")
        object.__setattr__(self, "prompt_batches", batches)
        # Validate every controller-owned frame before any OCI/GPU resource starts.
        try:
            probe_session = "1" * 32
            init = make_init(
                self.engine_config,
                session_id=probe_session,
                launch_digest=self.launch_digest,
                expected_engine_config_digest=self.expected_engine_config_digest,
            )
            frame_message(init, max_bytes=MAX_INIT_BYTES)
            accept = preflight_accept_message(
                session_id=probe_session,
                launch_digest=self.launch_digest,
                facts=self.expected_preflight,
            )
            frame_message(accept, max_bytes=MAX_CONTROL_BYTES)
        except SessionProtocolError as exc:
            raise OuterSessionInfrastructureError(
                f"controller init violates protocol policy: {exc}"
            ) from None
        for index, prompts in enumerate(batches):
            try:
                message = batch_request(
                    session_id="1" * 32,
                    launch_digest=self.launch_digest,
                    request_id="2" * 32,
                    nonce="3" * 32,
                    batch_index=index,
                    prompts=prompts,
                    max_new_tokens=self.max_new_tokens,
                    top_logprobs_num=self.top_logprobs_num,
                    temperature=self.temperature,
                )
                frame_message(message, max_bytes=MAX_BATCH_REQUEST_BYTES)
            except SessionProtocolError as exc:
                raise OuterSessionInfrastructureError(
                    f"controller batch {index} violates protocol policy: {exc}"
                ) from None


@dataclass(frozen=True)
class BatchExecutionEvidence:
    batch_index: int
    request_id: str
    nonce: str
    request_started_at: float
    response_completed_at: float
    token_numerator: int
    evidence: BatchEvidence

    @property
    def elapsed_seconds(self) -> float:
        return self.response_completed_at - self.request_started_at


@dataclass(frozen=True)
class SessionExecutionEvidence:
    session_id: str
    launch_digest: str
    preflight: RuntimePreflightFacts
    ready_completed_at: float
    batches: tuple[BatchExecutionEvidence, ...]
    warmup_count: int
    conditioning_count: int
    conditioning_started_at: float
    first_timed_completed_at: float
    conditioning_token_numerator: int
    session_completed_at: float
    discovery_activation: DiscoveryActivationReceipt | None = None

    @property
    def conditioning_interval_seconds(self) -> float:
        return self.first_timed_completed_at - self.conditioning_started_at


BoundaryCallback = Callable[[str, int, float], None]


def diagnostic_provider(
    transport: object,
) -> Callable[[], OCIAttachedDiagnostic] | None:
    """Return only the typed host diagnostic accessor exposed by a transport."""

    provider = getattr(transport, "stderr_diagnostic", None)
    return provider if callable(provider) else None


def _now(clock: Callable[[], float], *, previous: float | None = None) -> float:
    try:
        value = float(clock())
    except Exception as exc:
        raise OuterSessionInfrastructureError(f"host clock failed: {exc}") from None
    if not math.isfinite(value) or (previous is not None and value < previous):
        raise OuterSessionInfrastructureError("host clock moved backwards")
    return value


def _control_or_error(
    transport: SessionTransport,
    *,
    session_id: str,
    launch_digest: str,
    deadline: float,
) -> dict:
    message = transport.read_control(max_bytes=MAX_CONTROL_BYTES, deadline=deadline)
    try:
        detail = parse_error_message(
            message, session_id=session_id, launch_digest=launch_digest
        )
    except SessionProtocolError as exc:
        raise OuterSessionProtocolError(str(exc)) from None
    if detail is not None:
        raise OuterSessionWorkerError(
            ": ".join(detail), diagnostic_provider(transport)
        )
    return message


def _fresh_id(seen: set[str]) -> str:
    value = secrets.token_hex(16)
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(char not in "0123456789abcdef" for char in value)
        or value == "0" * 32
        or value in seen
    ):
        raise OuterSessionInfrastructureError("system RNG repeated a session binding")
    seen.add(value)
    return value


def run_outer_session(
    plan: SessionExecutionPlan,
    *,
    transport: SessionTransport,
    deadline: float,
    init_timeout_s: float,
    batch_timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
    boundary_callback: BoundaryCallback | None = None,
) -> SessionExecutionEvidence:
    """Execute one session and return host-timed raw facts, then destroy it."""

    if not isinstance(plan, SessionExecutionPlan):
        raise OuterSessionInfrastructureError("session plan is not typed")
    started_at = _now(clock)
    for name, value in (
        ("deadline", deadline),
        ("init_timeout_s", init_timeout_s),
        ("batch_timeout_s", batch_timeout_s),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or (name == "deadline" and value <= started_at)
            or (name != "deadline" and value <= 0)
        ):
            raise OuterSessionInfrastructureError(f"{name} is invalid")
    deadline = float(deadline)

    def phase_deadline(limit: float) -> float:
        return min(deadline, _now(clock) + float(limit))

    seen: set[str] = set()
    session_id = _fresh_id(seen)
    batch_rows: list[BatchExecutionEvidence] = []
    conditioning_start_index = plan.warmup_count - plan.conditioning_count
    conditioning_started_at: float | None = None
    first_timed_completed_at: float | None = None
    preflight: RuntimePreflightFacts | None = None
    ready_completed_at = 0.0
    try:
        transport.start()
        if transport.has_pending_output():
            raise OuterSessionProtocolError("worker emitted output before init")
        init_deadline = phase_deadline(init_timeout_s)
        init = make_init(
            plan.engine_config,
            session_id=session_id,
            launch_digest=plan.launch_digest,
            expected_engine_config_digest=plan.expected_engine_config_digest,
        )
        transport.write_frame(
            frame_message(init, max_bytes=MAX_INIT_BYTES), deadline=init_deadline
        )
        try:
            preflight = validate_preflight(
                _control_or_error(
                    transport,
                    session_id=session_id,
                    launch_digest=plan.launch_digest,
                    deadline=init_deadline,
                ),
                session_id=session_id,
                launch_digest=plan.launch_digest,
                expected_facts=plan.expected_preflight,
            )
        except (SessionProtocolError, OuterSessionProtocolError, OuterSessionWorkerError) as exc:
            detail = exc.message if isinstance(exc, OuterSessionError) else str(exc)
            raise OuterSessionInfrastructureError(
                f"runtime preflight failed: {detail}",
                diagnostic_provider(transport),
            ) from None
        transport.write_frame(
            frame_message(
                preflight_accept_message(
                    session_id=session_id,
                    launch_digest=plan.launch_digest,
                    facts=preflight,
                ),
                max_bytes=MAX_CONTROL_BYTES,
            ),
            deadline=init_deadline,
        )
        ready = _control_or_error(
            transport,
            session_id=session_id,
            launch_digest=plan.launch_digest,
            deadline=init_deadline,
        )
        try:
            expected_identity = plan.expected_discovery_overlay_identity_digest
            discovery_activation = validate_ready(
                ready,
                session_id=session_id,
                launch_digest=plan.launch_digest,
                expected_discovery_identity_digest=expected_identity,
                expected_discovery_tp_size=(
                    plan.engine_config.tp_size if expected_identity is not None else None
                ),
                expected_discovery_sglang_version=(
                    plan.expected_preflight.sglang_version
                    if expected_identity is not None
                    else None
                ),
            )
        except SessionProtocolError as exc:
            raise OuterSessionProtocolError(str(exc)) from None
        ready_completed_at = _now(clock, previous=started_at)
        last_host_time = ready_completed_at
        if conditioning_start_index == 0:
            conditioning_started_at = ready_completed_at
        if transport.has_pending_output():
            raise OuterSessionProtocolError("worker emitted output before first request")

        for index, prompts in enumerate(plan.prompt_batches):
            request_id, nonce = _fresh_id(seen), _fresh_id(seen)
            request = validate_batch_request(batch_request(
                session_id=session_id,
                launch_digest=plan.launch_digest,
                request_id=request_id,
                nonce=nonce,
                batch_index=index,
                prompts=prompts,
                max_new_tokens=plan.max_new_tokens,
                top_logprobs_num=plan.top_logprobs_num,
                temperature=plan.temperature,
            ))
            final_warmup = index == plan.warmup_count - 1
            first_timed = index == plan.warmup_count
            if final_warmup and boundary_callback is not None:
                boundary_callback("before_final_warmup", index, deadline)
            if first_timed and boundary_callback is not None:
                boundary_callback("before_first_timed", index, deadline)
            if transport.has_pending_output():
                raise OuterSessionProtocolError("worker emitted early or duplicate output")
            batch_deadline = phase_deadline(batch_timeout_s)
            request_started = _now(clock, previous=last_host_time)
            transport.write_frame(
                frame_message(request.to_dict(), max_bytes=MAX_BATCH_REQUEST_BYTES),
                deadline=batch_deadline,
            )
            evidence = transport.read_evidence(request, deadline=batch_deadline)
            completed = _now(clock, previous=request_started)
            if completed <= request_started:
                raise OuterSessionInfrastructureError("host batch clock did not advance")
            last_host_time = completed
            token_numerator = len(prompts) * plan.max_new_tokens
            if evidence.observed_tokens != token_numerator:
                raise OuterSessionProtocolError("worker evidence token count is not exact")
            batch_rows.append(BatchExecutionEvidence(
                index, request_id, nonce, request_started, completed,
                token_numerator, evidence,
            ))
            if index + 1 == conditioning_start_index:
                conditioning_started_at = completed
            if final_warmup and boundary_callback is not None:
                boundary_callback("after_final_warmup", index, deadline)
            if first_timed:
                first_timed_completed_at = completed
            if transport.has_pending_output():
                raise OuterSessionProtocolError("worker emitted trailing or duplicate output")

        # There is no close frame.  The host destroys the engine immediately after
        # the final exact response and the manager proves process/container absence.
        if _now(clock, previous=last_host_time) >= deadline:
            raise OuterSessionTimeoutError("session deadline expired before cleanup")
        transport.finalize()
        session_completed_at = _now(clock, previous=batch_rows[-1].response_completed_at)
        if session_completed_at > deadline:
            raise OuterSessionTimeoutError("session cleanup exceeded its absolute deadline")
    except BaseException as original:
        if isinstance(original, OuterSessionError):
            original.attach_diagnostic(diagnostic_provider(transport))
        try:
            transport.abort()
        except BaseException as cleanup:
            error = OuterSessionInfrastructureError(
                f"session cleanup could not be proven: {cleanup}"
            )
            error.attach_diagnostic(diagnostic_provider(transport))
            raise error from original
        raise

    if preflight is None or conditioning_started_at is None or first_timed_completed_at is None:
        raise OuterSessionInfrastructureError("session lacks required conditioning evidence")
    conditioning_end = plan.warmup_count
    conditioning_tokens = sum(
        row.token_numerator
        for row in batch_rows[conditioning_start_index : conditioning_end + 1]
    )
    if first_timed_completed_at <= conditioning_started_at:
        raise OuterSessionInfrastructureError("conditioning interval did not advance")
    return SessionExecutionEvidence(
        session_id=session_id,
        launch_digest=plan.launch_digest,
        preflight=preflight,
        ready_completed_at=ready_completed_at,
        batches=tuple(batch_rows),
        warmup_count=plan.warmup_count,
        conditioning_count=plan.conditioning_count,
        conditioning_started_at=conditioning_started_at,
        first_timed_completed_at=first_timed_completed_at,
        conditioning_token_numerator=conditioning_tokens,
        session_completed_at=session_completed_at,
        discovery_activation=discovery_activation,
    )
