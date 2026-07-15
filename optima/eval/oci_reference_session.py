"""Trusted host protocol for one pristine, untimed teacher session.

The caller supplies an exact empty-stack reference manifest and an ordered cohort
of fixed-width requests.  This module performs the ordinary OCI init/preflight
handshake, exchanges only ORQ1/ORE1 frames, and returns primitive worker evidence
plus host-observed transcript identities.  It has no quality, scheduling, or
economic authority.
"""

from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from optima.eval.oci_outer_session import (
    AttachedSessionTransport,
    OuterSessionError,
    OuterSessionInfrastructureError,
    OuterSessionProtocolError,
    OuterSessionTimeoutError,
    OuterSessionWorkerError,
    diagnostic_provider,
)
from optima.eval.oci_session_protocol import (
    CONTROL_MAGIC,
    MAX_CONTROL_BYTES,
    MAX_INIT_BYTES,
    EngineSessionConfig,
    RuntimePreflightFacts,
    SessionProtocolError,
    decode_message,
    frame_message,
    make_init,
    parse_error_message,
    preflight_accept_message,
    validate_preflight,
    validate_ready,
)
from optima.eval.qualification import ReferenceManifest
from optima.eval.reference_protocol import (
    EVIDENCE_MAGIC,
    MAX_EVIDENCE_BYTES,
    ReferenceEvidence,
    ReferenceProtocolError,
    ReferenceRequest,
    decode_reference_evidence,
    encode_reference_request,
    encode_reference_evidence,
    expected_evidence_payload_bytes,
    request_sha256,
)
from optima.stack_identity import canonical_digest, require_sha256_hex, sha256_hex
from optima.stack_manifest import EvaluationStackManifest


class ReferenceSessionError(RuntimeError):
    """The pristine-reference plan or transcript is invalid."""


class ReferenceTransport(Protocol):
    def start(self) -> None: ...
    def has_pending_output(self) -> bool: ...
    def write_frame(self, frame: bytes, *, deadline: float) -> None: ...
    def read_control(self, *, max_bytes: int, deadline: float) -> dict: ...
    def read_reference_evidence(
        self, request: ReferenceRequest, *, deadline: float
    ) -> ReferenceEvidence: ...
    def finalize(self) -> None: ...
    def abort(self) -> None: ...


def _digest(value: object, *, field: str) -> str:
    try:
        result = require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise ReferenceSessionError(str(exc)) from None
    if result == "0" * 64:
        raise ReferenceSessionError(f"{field} must not be the all-zero digest")
    return result


def _now(clock: Callable[[], float], *, previous: float | None = None) -> float:
    try:
        result = float(clock())
    except Exception as exc:
        raise OuterSessionInfrastructureError(f"host clock failed: {exc}") from None
    if not math.isfinite(result) or (previous is not None and result < previous):
        raise OuterSessionInfrastructureError("host clock moved backwards")
    return result


@dataclass(frozen=True)
class ReferenceSessionPlan:
    """Exact empty-stack launch and ordered ORQ1 cohort supplied by the validator."""

    reference: ReferenceManifest
    pristine_stack: EvaluationStackManifest
    expected_engine_config_digest: str
    engine_config: EngineSessionConfig
    expected_preflight: RuntimePreflightFacts
    request_plan_digest: str
    requests: tuple[ReferenceRequest, ...]

    def __post_init__(self) -> None:
        if type(self.reference) is not ReferenceManifest:
            raise ReferenceSessionError("reference manifest is not typed")
        if type(self.pristine_stack) is not EvaluationStackManifest:
            raise ReferenceSessionError("pristine stack is not typed")
        if self.pristine_stack.entries:
            raise ReferenceSessionError("pristine reference stack contains contributions")
        if (
            self.reference.pristine_stack_digest != self.pristine_stack.digest
            or self.reference.catalog_digest != self.pristine_stack.catalog_digest
            or (
                self.reference.runtime_digest,
                self.reference.base_engine_digest,
                self.reference.arena_digest,
            )
            != (
                self.pristine_stack.runtime_digest,
                self.pristine_stack.base_engine_digest,
                self.pristine_stack.arena_digest,
            )
        ):
            raise ReferenceSessionError("reference manifest differs from its empty stack")
        if type(self.engine_config) is not EngineSessionConfig:
            raise ReferenceSessionError("reference engine config is not typed")
        expected_config = _digest(
            self.expected_engine_config_digest,
            field="expected_engine_config_digest",
        )
        object.__setattr__(self, "expected_engine_config_digest", expected_config)
        if self.engine_config.digest != expected_config:
            raise ReferenceSessionError("reference engine config digest differs from plan")
        if (
            type(self.expected_preflight) is not RuntimePreflightFacts
            or self.expected_preflight.launch_digest
            != self.reference.pristine_launch_digest
            or self.expected_preflight.engine_config_digest != expected_config
        ):
            raise ReferenceSessionError("reference preflight differs from plan identity")
        request_plan = _digest(self.request_plan_digest, field="request_plan_digest")
        object.__setattr__(self, "request_plan_digest", request_plan)
        if type(self.requests) is not tuple or not self.requests or any(
            type(row) is not ReferenceRequest for row in self.requests
        ):
            raise ReferenceSessionError("reference requests must be a nonempty typed tuple")
        session_id = self.requests[0].session_id
        expected_indices = tuple(range(len(self.requests)))
        if tuple(row.request_index for row in self.requests) != expected_indices:
            raise ReferenceSessionError("reference request indices are not contiguous")
        if any(
            (
                row.session_id != session_id
                or row.launch_digest != self.reference.pristine_launch_digest
                or row.plan_digest != request_plan
            )
            for row in self.requests
        ):
            raise ReferenceSessionError("reference request cohort binding differs from plan")
        bindings = [session_id]
        bindings.extend(row.request_id for row in self.requests)
        bindings.extend(row.nonce for row in self.requests)
        if len(bindings) != len(set(bindings)):
            raise ReferenceSessionError("reference request cohort repeats a session binding")

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.eval.reference-session-plan",
            {
                "engine_config_digest": self.engine_config.digest,
                "expected_preflight_digest": self.expected_preflight.digest,
                "pristine_stack_digest": self.pristine_stack.digest,
                "reference_manifest_digest": self.reference.digest,
                "request_plan_digest": self.request_plan_digest,
                "request_sha256": [request_sha256(row) for row in self.requests],
            },
        )


@dataclass(frozen=True)
class ReferenceExchangeEvidence:
    request_index: int
    request: ReferenceRequest
    request_sha256: str
    evidence_frame_sha256: str
    request_started_at: float
    response_completed_at: float
    evidence: ReferenceEvidence

    def __post_init__(self) -> None:
        if type(self.request_index) is not int or self.request_index < 0:
            raise ReferenceSessionError("reference exchange index is invalid")
        if type(self.request) is not ReferenceRequest:
            raise ReferenceSessionError("reference exchange request is not typed")
        object.__setattr__(
            self, "request_sha256", _digest(self.request_sha256, field="request_sha256")
        )
        object.__setattr__(
            self,
            "evidence_frame_sha256",
            _digest(self.evidence_frame_sha256, field="evidence_frame_sha256"),
        )
        try:
            expected_response = sha256_hex(
                encode_reference_evidence(self.evidence, self.request)
            )
        except ReferenceProtocolError as exc:
            raise ReferenceSessionError(f"reference exchange binding failed: {exc}") from None
        if (
            type(self.request_started_at) is not float
            or type(self.response_completed_at) is not float
            or not math.isfinite(self.request_started_at)
            or not math.isfinite(self.response_completed_at)
            or self.response_completed_at <= self.request_started_at
            or type(self.evidence) is not ReferenceEvidence
            or self.request.request_index != self.request_index
            or request_sha256(self.request) != self.request_sha256
            or self.evidence.request_index != self.request_index
            or self.evidence.request_sha256 != self.request_sha256
            or expected_response != self.evidence_frame_sha256
        ):
            raise ReferenceSessionError("reference exchange evidence is malformed")


@dataclass(frozen=True)
class ReferenceSessionEvidence:
    schema: str
    session_id: str
    launch_digest: str
    reference_manifest_digest: str
    session_plan_digest: str
    request_plan_digest: str
    preflight: RuntimePreflightFacts
    ready_completed_at: float
    exchanges: tuple[ReferenceExchangeEvidence, ...]
    session_completed_at: float

    def __post_init__(self) -> None:
        if self.schema != "optima.pristine-reference-session.v1":
            raise ReferenceSessionError("reference session schema is invalid")
        for field in (
            "launch_digest",
            "reference_manifest_digest",
            "session_plan_digest",
            "request_plan_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        if (
            not isinstance(self.session_id, str)
            or len(self.session_id) != 32
            or any(char not in "0123456789abcdef" for char in self.session_id)
            or type(self.preflight) is not RuntimePreflightFacts
            or self.preflight.launch_digest != self.launch_digest
            or type(self.exchanges) is not tuple
            or not self.exchanges
            or any(type(row) is not ReferenceExchangeEvidence for row in self.exchanges)
            or tuple(row.request_index for row in self.exchanges)
            != tuple(range(len(self.exchanges)))
            or any(
                row.request.session_id != self.session_id
                or row.request.launch_digest != self.launch_digest
                or row.request.plan_digest != self.request_plan_digest
                or row.evidence.session_id != self.session_id
                or row.evidence.launch_digest != self.launch_digest
                or row.evidence.plan_digest != self.request_plan_digest
                for row in self.exchanges
            )
            or type(self.ready_completed_at) is not float
            or type(self.session_completed_at) is not float
            or not math.isfinite(self.ready_completed_at)
            or not math.isfinite(self.session_completed_at)
            or self.exchanges[0].request_started_at < self.ready_completed_at
            or any(
                current.response_completed_at > following.request_started_at
                for current, following in zip(
                    self.exchanges, self.exchanges[1:]
                )
            )
            or self.session_completed_at <= self.exchanges[-1].response_completed_at
        ):
            raise ReferenceSessionError("reference session evidence is malformed")

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.eval.pristine-reference-session",
            {
                "exchanges": [
                    {
                        "request_index": row.request_index,
                        "request_sha256": row.request_sha256,
                        "evidence_frame_sha256": row.evidence_frame_sha256,
                    }
                    for row in self.exchanges
                ],
                "launch_digest": self.launch_digest,
                "preflight_digest": self.preflight.digest,
                "reference_manifest_digest": self.reference_manifest_digest,
                "request_plan_digest": self.request_plan_digest,
                "schema": self.schema,
                "session_id": self.session_id,
                "session_plan_digest": self.session_plan_digest,
            },
        )


class AttachedReferenceTransport(AttachedSessionTransport):
    """ORQ1/ORE1 reader over the ordinary manager-owned attached transport."""

    def read_reference_evidence(
        self, request: ReferenceRequest, *, deadline: float
    ) -> ReferenceEvidence:
        magic, size = self._header(deadline=deadline)
        if magic == CONTROL_MAGIC:
            if size > MAX_CONTROL_BYTES:
                raise OuterSessionProtocolError("worker declared an oversized error frame")
            try:
                message = decode_message(
                    self._read_exact(size, deadline=deadline),
                    max_bytes=MAX_CONTROL_BYTES,
                )
                detail = parse_error_message(
                    message,
                    session_id=request.session_id,
                    launch_digest=request.launch_digest,
                )
            except SessionProtocolError as exc:
                raise OuterSessionProtocolError(str(exc)) from None
            if detail is not None:
                raise self._diagnostic_error(
                    OuterSessionWorkerError, ": ".join(detail)
                )
            raise OuterSessionProtocolError("worker emitted an early control frame")
        if magic != EVIDENCE_MAGIC:
            raise OuterSessionProtocolError("worker emitted wrong reference-evidence magic")
        exact = expected_evidence_payload_bytes(request)
        if size != exact or size > MAX_EVIDENCE_BYTES:
            raise OuterSessionProtocolError("worker reference evidence has the wrong exact size")
        payload = self._read_exact(size, deadline=deadline)
        try:
            return decode_reference_evidence(
                EVIDENCE_MAGIC + struct.pack(">I", size) + payload,
                request,
            )
        except ReferenceProtocolError as exc:
            raise OuterSessionProtocolError(str(exc)) from None


def _control_or_error(
    transport: ReferenceTransport,
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


def run_reference_session(
    plan: ReferenceSessionPlan,
    *,
    transport: ReferenceTransport,
    deadline: float,
    init_timeout_s: float,
    batch_timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> ReferenceSessionEvidence:
    """Run one ordered finalist cohort through one pristine teacher lifetime."""

    if type(plan) is not ReferenceSessionPlan:
        raise ReferenceSessionError("reference session plan is not typed")
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
            or float(value) <= (started_at if name == "deadline" else 0.0)
        ):
            raise OuterSessionInfrastructureError(f"{name} is invalid")
    deadline = float(deadline)

    def phase_deadline(limit: float) -> float:
        return min(deadline, _now(clock) + float(limit))

    session_id = plan.requests[0].session_id
    launch_digest = plan.reference.pristine_launch_digest
    preflight: RuntimePreflightFacts | None = None
    exchanges: list[ReferenceExchangeEvidence] = []
    try:
        transport.start()
        if transport.has_pending_output():
            raise OuterSessionProtocolError("reference worker emitted output before init")
        init_deadline = phase_deadline(init_timeout_s)
        init = make_init(
            plan.engine_config,
            session_id=session_id,
            launch_digest=launch_digest,
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
                    launch_digest=launch_digest,
                    deadline=init_deadline,
                ),
                session_id=session_id,
                launch_digest=launch_digest,
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
                    session_id=session_id, launch_digest=launch_digest, facts=preflight
                ),
                max_bytes=MAX_CONTROL_BYTES,
            ),
            deadline=init_deadline,
        )
        ready = _control_or_error(
            transport,
            session_id=session_id,
            launch_digest=launch_digest,
            deadline=init_deadline,
        )
        try:
            validate_ready(ready, session_id=session_id, launch_digest=launch_digest)
        except SessionProtocolError as exc:
            raise OuterSessionProtocolError(str(exc)) from None
        ready_at = _now(clock, previous=started_at)
        last_time = ready_at
        if transport.has_pending_output():
            raise OuterSessionProtocolError("reference worker emitted output before first request")
        for request in plan.requests:
            if transport.has_pending_output():
                raise OuterSessionProtocolError("reference worker emitted early or duplicate output")
            request_deadline = phase_deadline(batch_timeout_s)
            request_started = _now(clock, previous=last_time)
            transport.write_frame(
                encode_reference_request(request), deadline=request_deadline
            )
            raw = transport.read_reference_evidence(request, deadline=request_deadline)
            completed = _now(clock, previous=request_started)
            if completed <= request_started:
                raise OuterSessionInfrastructureError("host reference request clock did not advance")
            exchanges.append(
                ReferenceExchangeEvidence(
                    request.request_index,
                    request,
                    request_sha256(request),
                    sha256_hex(encode_reference_evidence(raw, request)),
                    request_started,
                    completed,
                    raw,
                )
            )
            last_time = completed
            if transport.has_pending_output():
                raise OuterSessionProtocolError("reference worker emitted trailing output")
        if _now(clock, previous=last_time) >= deadline:
            raise OuterSessionTimeoutError("reference deadline expired before cleanup")
        transport.finalize()
        completed_at = _now(clock, previous=last_time)
        if completed_at > deadline:
            raise OuterSessionTimeoutError("reference cleanup exceeded its absolute deadline")
    except BaseException as original:
        if isinstance(original, OuterSessionError):
            original.attach_diagnostic(diagnostic_provider(transport))
        try:
            transport.abort()
        except BaseException as cleanup:
            error = OuterSessionInfrastructureError(
                f"reference cleanup could not be proven: {cleanup}"
            )
            error.attach_diagnostic(diagnostic_provider(transport))
            raise error from original
        raise
    if preflight is None:  # pragma: no cover - successful handshake sets it
        raise OuterSessionInfrastructureError("reference session lacks preflight evidence")
    return ReferenceSessionEvidence(
        "optima.pristine-reference-session.v1",
        session_id,
        launch_digest,
        plan.reference.digest,
        plan.digest,
        plan.request_plan_digest,
        preflight,
        ready_at,
        tuple(exchanges),
        completed_at,
    )
