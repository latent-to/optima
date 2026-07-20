"""Trusted host controller for one resident (hot-swap) engine lifetime.

A resident session keeps ONE stock-launched engine alive while an ordered verb
stream swaps candidate bundles in and out of the live dispatch registry (CUDA
graphs recapture per swap; weights never move) and executes timed prompt
batches between swaps.  Every batch's evidence names the swap generation that
was live, so "which kernel was scored" stays provable from the ordered stream.

Trust tier: resident evidence routes and screens; it is NOT payment/crown
evidence.  A hostile candidate that has executed inside the engine could in
principle contaminate later reads in the same lifetime — the queue layer
mitigates with stock canaries, recycle policy, and re-screens, and the isolated
per-candidate path remains the settlement authority.  This module owns only
framed byte I/O and the host clock, exactly like :mod:`oci_outer_session`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, NoReturn, Sequence

from optima.eval.oci_outer_session import (
    OuterSessionError,
    OuterSessionInfrastructureError,
    OuterSessionProtocolError,
    SessionTransport,
    _control_or_error,
    _fresh_id,
    _now,
    diagnostic_provider,
)
from optima.eval.oci_session_protocol import (
    MAX_BATCH_REQUEST_BYTES,
    MAX_CONTROL_BYTES,
    MAX_INIT_BYTES,
    BatchEvidence,
    EngineSessionConfig,
    RuntimePreflightFacts,
    SessionProtocolError,
    SwapRequest,
    batch_request,
    frame_message,
    make_init,
    preflight_accept_message,
    validate_batch_request,
    validate_preflight,
    validate_ready,
    validate_swap_evidence,
)
from optima.stack_identity import require_sha256_hex


MAX_RESIDENT_SWAPS = 10_000
MAX_RESIDENT_BATCHES = 100_000


@dataclass(frozen=True)
class ResidentSessionPlan:
    """Host-only inputs for one resident lifetime; verbs arrive incrementally.

    Unlike :class:`SessionExecutionPlan` there is no fixed prompt-batch list:
    the queue layer issues swaps and reads as it schedules candidates.  Every
    outgoing frame is still fully validated at issue time by the protocol
    constructors, so the closed-world property holds per verb.
    """

    launch_digest: str
    expected_engine_config_digest: str
    engine_config: EngineSessionConfig
    expected_preflight: RuntimePreflightFacts
    max_swaps: int
    max_batches: int
    max_new_tokens: int
    top_logprobs_num: int
    temperature: float

    def __post_init__(self) -> None:
        if not isinstance(self.engine_config, EngineSessionConfig):
            raise OuterSessionInfrastructureError("engine_config is not typed")
        if self.engine_config.digest != self.expected_engine_config_digest:
            raise OuterSessionInfrastructureError(
                "engine config digest differs from plan"
            )
        if (
            not isinstance(self.expected_preflight, RuntimePreflightFacts)
            or self.expected_preflight.launch_digest != self.launch_digest
            or self.expected_preflight.engine_config_digest
            != self.expected_engine_config_digest
        ):
            raise OuterSessionInfrastructureError(
                "preflight facts differ from plan identity"
            )
        for name, value, low, high in (
            ("max_swaps", self.max_swaps, 0, MAX_RESIDENT_SWAPS),
            ("max_batches", self.max_batches, 1, MAX_RESIDENT_BATCHES),
        ):
            if type(value) is not int or not low <= value <= high:
                raise OuterSessionInfrastructureError(f"{name} is invalid")
        # Validate the handshake frames before any OCI/GPU resource starts.
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
        # Validate the fixed read shape once with a representative one-prompt
        # batch so an unschedulable plan fails before the engine loads.
        try:
            batch_request(
                session_id="1" * 32,
                launch_digest=self.launch_digest,
                request_id="2" * 32,
                nonce="3" * 32,
                batch_index=0,
                prompts=("probe",),
                max_new_tokens=self.max_new_tokens,
                top_logprobs_num=self.top_logprobs_num,
                temperature=self.temperature,
            )
        except SessionProtocolError as exc:
            raise OuterSessionInfrastructureError(
                f"controller read shape violates protocol policy: {exc}"
            ) from None


@dataclass(frozen=True)
class SwapReceipt:
    """Host-clock record of one applied swap on a live resident engine."""

    swap_index: int
    generation: int
    bundle_digest: str | None
    slots: tuple[str, ...]
    requested_at: float
    completed_at: float

    def __post_init__(self) -> None:
        if (
            type(self.swap_index) is not int
            or self.swap_index < 0
            or type(self.generation) is not int
            or self.generation < 1
        ):
            raise OuterSessionInfrastructureError("swap receipt ordering is invalid")
        if self.bundle_digest is not None:
            try:
                require_sha256_hex(self.bundle_digest, field="swap bundle digest")
            except ValueError as exc:
                raise OuterSessionInfrastructureError(str(exc)) from None
        if (self.bundle_digest is None) != (not self.slots):
            raise OuterSessionInfrastructureError(
                "swap receipt slots contradict its bundle identity"
            )
        for value in (self.requested_at, self.completed_at):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise OuterSessionInfrastructureError("swap receipt clock is invalid")
        if self.completed_at <= self.requested_at:
            raise OuterSessionInfrastructureError("swap receipt clock did not advance")

    @property
    def swap_seconds(self) -> float:
        return self.completed_at - self.requested_at

    def to_dict(self) -> dict[str, object]:
        return {
            "bundle_digest": self.bundle_digest,
            "completed_at": format(self.completed_at, ".17g"),
            "generation": self.generation,
            "requested_at": format(self.requested_at, ".17g"),
            "slots": list(self.slots),
            "swap_index": self.swap_index,
        }


@dataclass(frozen=True)
class ResidentBatchEvidence:
    """One timed read bound to the swap generation that was live."""

    batch_index: int
    request_id: str
    nonce: str
    generation: int
    active_slots: tuple[str, ...]
    canary: bool
    request_started_at: float
    response_completed_at: float
    token_numerator: int
    evidence: BatchEvidence

    @property
    def elapsed_seconds(self) -> float:
        return self.response_completed_at - self.request_started_at


@dataclass(frozen=True)
class ResidentSessionEvidence:
    """Raw ordered stream evidence for one resident engine lifetime."""

    session_id: str
    launch_digest: str
    preflight: RuntimePreflightFacts
    ready_completed_at: float
    batches: tuple[ResidentBatchEvidence, ...]
    swaps: tuple[SwapReceipt, ...]
    session_completed_at: float


class ResidentOuterSession:
    """Incremental trusted-host controller for one resident engine lifetime.

    The verb stream is strictly serialized over one pipe: a batch executed
    after ``swap(generation=G)`` ran under generation G by construction.  The
    caller (queue layer) owns candidate identity, scoring, and recycle policy.
    """

    def __init__(
        self,
        plan: ResidentSessionPlan,
        *,
        transport: SessionTransport,
        deadline: float,
        init_timeout_s: float,
        batch_timeout_s: float,
        swap_timeout_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(plan) is not ResidentSessionPlan:
            raise OuterSessionInfrastructureError("resident plan is not typed")
        started_at = _now(clock)
        for name, value in (
            ("deadline", deadline),
            ("init_timeout_s", init_timeout_s),
            ("batch_timeout_s", batch_timeout_s),
            ("swap_timeout_s", swap_timeout_s),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or (name == "deadline" and value <= started_at)
                or (name != "deadline" and value <= 0)
            ):
                raise OuterSessionInfrastructureError(f"{name} is invalid")
        self.plan = plan
        self.transport = transport
        self.deadline = float(deadline)
        self.init_timeout_s = float(init_timeout_s)
        self.batch_timeout_s = float(batch_timeout_s)
        self.swap_timeout_s = float(swap_timeout_s)
        self.clock = clock
        self.started_at = started_at
        self.seen: set[str] = set()
        self.session_id = _fresh_id(self.seen)
        self.batch_rows: list[ResidentBatchEvidence] = []
        self.swap_receipts: list[SwapReceipt] = []
        self.active_generation = 0
        self.active_slots: tuple[str, ...] = ()
        self.active_bundle_digest: str | None = None
        self.preflight: RuntimePreflightFacts | None = None
        self.ready_completed_at = 0.0
        self.last_host_time = started_at
        self.started = False
        self.closed = False

    def _phase_deadline(self, limit: float) -> float:
        return min(self.deadline, _now(self.clock) + limit)

    def _fail(self, original: BaseException) -> NoReturn:
        if isinstance(original, OuterSessionError):
            original.attach_diagnostic(diagnostic_provider(self.transport))
        try:
            self.transport.abort()
        except BaseException as cleanup:
            error = OuterSessionInfrastructureError(
                f"session cleanup could not be proven: {cleanup}"
            )
            error.attach_diagnostic(diagnostic_provider(self.transport))
            self.closed = True
            raise error from original
        self.closed = True
        raise original

    def start(self) -> None:
        if self.started or self.closed:
            raise OuterSessionInfrastructureError("session start order is invalid")
        try:
            self.transport.start()
            if self.transport.has_pending_output():
                raise OuterSessionProtocolError("worker emitted output before init")
            init_deadline = self._phase_deadline(self.init_timeout_s)
            init = make_init(
                self.plan.engine_config,
                session_id=self.session_id,
                launch_digest=self.plan.launch_digest,
                expected_engine_config_digest=(
                    self.plan.expected_engine_config_digest
                ),
            )
            self.transport.write_frame(
                frame_message(init, max_bytes=MAX_INIT_BYTES), deadline=init_deadline
            )
            try:
                self.preflight = validate_preflight(
                    _control_or_error(
                        self.transport,
                        session_id=self.session_id,
                        launch_digest=self.plan.launch_digest,
                        deadline=init_deadline,
                    ),
                    session_id=self.session_id,
                    launch_digest=self.plan.launch_digest,
                    expected_facts=self.plan.expected_preflight,
                )
            except (SessionProtocolError, OuterSessionError) as exc:
                detail = (
                    exc.message if isinstance(exc, OuterSessionError) else str(exc)
                )
                raise OuterSessionInfrastructureError(
                    f"runtime preflight failed: {detail}",
                    diagnostic_provider(self.transport),
                ) from None
            self.transport.write_frame(
                frame_message(
                    preflight_accept_message(
                        session_id=self.session_id,
                        launch_digest=self.plan.launch_digest,
                        facts=self.preflight,
                    ),
                    max_bytes=MAX_CONTROL_BYTES,
                ),
                deadline=init_deadline,
            )
            ready = _control_or_error(
                self.transport,
                session_id=self.session_id,
                launch_digest=self.plan.launch_digest,
                deadline=init_deadline,
            )
            try:
                validate_ready(
                    ready,
                    session_id=self.session_id,
                    launch_digest=self.plan.launch_digest,
                )
            except SessionProtocolError as exc:
                raise OuterSessionProtocolError(str(exc)) from None
            self.ready_completed_at = _now(self.clock, previous=self.started_at)
            self.last_host_time = self.ready_completed_at
            if self.transport.has_pending_output():
                raise OuterSessionProtocolError(
                    "worker emitted output before first request"
                )
            self.started = True
        except BaseException as exc:
            self._fail(exc)

    def swap(self, bundle_digest: str | None) -> SwapReceipt:
        """Swap the live engine to a staged bundle (or back to stock)."""

        if not self.started or self.closed:
            raise OuterSessionInfrastructureError("session is not open")
        if len(self.swap_receipts) >= self.plan.max_swaps:
            raise OuterSessionInfrastructureError(
                "session exceeded its planned swap budget"
            )
        try:
            request_id, nonce = _fresh_id(self.seen), _fresh_id(self.seen)
            request = SwapRequest(
                self.session_id,
                self.plan.launch_digest,
                request_id,
                nonce,
                len(self.swap_receipts),
                self.active_generation + 1,
                bundle_digest,
            )
            if self.transport.has_pending_output():
                raise OuterSessionProtocolError(
                    "worker emitted early or duplicate output"
                )
            swap_deadline = self._phase_deadline(self.swap_timeout_s)
            requested_at = _now(self.clock, previous=self.last_host_time)
            self.transport.write_frame(
                frame_message(request.to_dict(), max_bytes=MAX_CONTROL_BYTES),
                deadline=swap_deadline,
            )
            try:
                slots = validate_swap_evidence(
                    _control_or_error(
                        self.transport,
                        session_id=self.session_id,
                        launch_digest=self.plan.launch_digest,
                        deadline=swap_deadline,
                    ),
                    request=request,
                    expected_rank_count=self.plan.engine_config.tp_size,
                )
            except SessionProtocolError as exc:
                raise OuterSessionProtocolError(str(exc)) from None
            completed_at = _now(self.clock, previous=requested_at)
            if self.transport.has_pending_output():
                raise OuterSessionProtocolError(
                    "worker emitted trailing or duplicate output"
                )
            receipt = SwapReceipt(
                request.swap_index,
                request.generation,
                request.bundle_digest,
                slots,
                requested_at,
                completed_at,
            )
            self.swap_receipts.append(receipt)
            self.active_generation = request.generation
            self.active_slots = slots
            self.active_bundle_digest = request.bundle_digest
            self.last_host_time = completed_at
            return receipt
        except BaseException as exc:
            self._fail(exc)

    def execute_batch(
        self, prompts: Sequence[str], *, canary: bool = False
    ) -> ResidentBatchEvidence:
        """Execute one timed read under the currently live generation."""

        if not self.started or self.closed:
            raise OuterSessionInfrastructureError("session is not open")
        if type(canary) is not bool:
            raise OuterSessionInfrastructureError("canary flag must be boolean")
        if len(self.batch_rows) >= self.plan.max_batches:
            raise OuterSessionInfrastructureError(
                "session exceeded its planned batch budget"
            )
        if canary and (self.active_slots or self.active_bundle_digest is not None):
            raise OuterSessionInfrastructureError(
                "canary reads require stock dispatch (swap to stock first)"
            )
        index = len(self.batch_rows)
        try:
            request_id, nonce = _fresh_id(self.seen), _fresh_id(self.seen)
            request = validate_batch_request(
                batch_request(
                    session_id=self.session_id,
                    launch_digest=self.plan.launch_digest,
                    request_id=request_id,
                    nonce=nonce,
                    batch_index=index,
                    prompts=tuple(prompts),
                    max_new_tokens=self.plan.max_new_tokens,
                    top_logprobs_num=self.plan.top_logprobs_num,
                    temperature=self.plan.temperature,
                )
            )
            if self.transport.has_pending_output():
                raise OuterSessionProtocolError(
                    "worker emitted early or duplicate output"
                )
            batch_deadline = self._phase_deadline(self.batch_timeout_s)
            request_started = _now(self.clock, previous=self.last_host_time)
            self.transport.write_frame(
                frame_message(request.to_dict(), max_bytes=MAX_BATCH_REQUEST_BYTES),
                deadline=batch_deadline,
            )
            evidence = self.transport.read_evidence(request, deadline=batch_deadline)
            completed = _now(self.clock, previous=request_started)
            if completed <= request_started:
                raise OuterSessionInfrastructureError(
                    "host batch clock did not advance"
                )
            token_numerator = len(request.prompts) * self.plan.max_new_tokens
            if evidence.observed_tokens != token_numerator:
                raise OuterSessionProtocolError(
                    "worker evidence token count is not exact"
                )
            if self.transport.has_pending_output():
                raise OuterSessionProtocolError(
                    "worker emitted trailing or duplicate output"
                )
            row = ResidentBatchEvidence(
                index,
                request_id,
                nonce,
                self.active_generation,
                self.active_slots,
                canary,
                request_started,
                completed,
                token_numerator,
                evidence,
            )
            self.batch_rows.append(row)
            self.last_host_time = completed
            return row
        except BaseException as exc:
            self._fail(exc)

    def finish(self) -> ResidentSessionEvidence:
        if not self.started or self.closed:
            raise OuterSessionInfrastructureError("session finish order is invalid")
        if not self.batch_rows:
            raise OuterSessionInfrastructureError(
                "resident session executed no batches"
            )
        try:
            if self.transport.has_pending_output():
                raise OuterSessionProtocolError(
                    "worker emitted trailing or duplicate output before cleanup"
                )
            self.transport.finalize()
            session_completed_at = _now(self.clock, previous=self.last_host_time)
        except BaseException as exc:
            self._fail(exc)
        self.closed = True
        if self.preflight is None:
            raise OuterSessionInfrastructureError(
                "resident session lacks preflight evidence"
            )
        return ResidentSessionEvidence(
            session_id=self.session_id,
            launch_digest=self.plan.launch_digest,
            preflight=self.preflight,
            ready_completed_at=self.ready_completed_at,
            batches=tuple(self.batch_rows),
            swaps=tuple(self.swap_receipts),
            session_completed_at=session_completed_at,
        )

    def abort(self) -> None:
        if self.closed:
            return
        try:
            self.transport.abort()
        finally:
            self.closed = True


__all__ = [
    "MAX_RESIDENT_BATCHES",
    "MAX_RESIDENT_SWAPS",
    "ResidentBatchEvidence",
    "ResidentOuterSession",
    "ResidentSessionEvidence",
    "ResidentSessionPlan",
    "SwapReceipt",
]
