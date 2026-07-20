"""Bounded production speed stage with two resident TP lanes.

Stock and candidate load once on disjoint lanes. GPU work is serialized because
simultaneous TP4 reads were measured to distort both lanes. The cheap B/C/B-prime
decision runs first; only a precommitted borderline result adds C-prime/B-double-
prime. Audit and pristine T are later stages and must not run unless this returns
PASS.
"""

from __future__ import annotations

import concurrent.futures
import math
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Callable

from optima.eval.engine_launch import EngineLaunchSpec, TrustedLaunchBinding
from optima.eval.oci_backend import (
    EngineExecutionEvidence,
    OCIEngineExecutor,
    TrustedArenaModelMountReceipt,
)
from optima.eval.oci_outer_session import (
    BatchExecutionEvidence,
    OpenedOuterSession,
    SessionExecutionEvidence,
    SessionExecutionPlan,
)
from optima.eval.oci_process import OCIQuiescenceReceipt
from optima.eval.scoring import SpeedupVerdict, marginal_workload_digest, score_speedup
from optima.stack_identity import canonical_digest, require_sha256_hex


class CrossoverRuntimeError(RuntimeError):
    pass


class SpeedStageDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NO_DECISION = "NO_DECISION"


@dataclass(frozen=True)
class ResidentSpeedPolicy:
    """Authority for adaptive reads and the speed-stage wall-clock SLA."""

    max_stage_seconds: int
    min_margin: float
    noise_multiplier: float
    max_noise: float
    calibration_digest: str
    calibration_context_digest: str
    version: int = 1
    max_qualification_seconds: int = 7_200

    def __post_init__(self) -> None:
        if (
            type(self.version) is not int
            or self.version != 1
            or type(self.max_stage_seconds) is not int
            or not 60 <= self.max_stage_seconds <= 7_200
            or type(self.max_qualification_seconds) is not int
            or not self.max_stage_seconds
            <= self.max_qualification_seconds
            <= 14_400
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in (
                    self.min_margin,
                    self.noise_multiplier,
                    self.max_noise,
                )
            )
            or not 0 < self.min_margin < 1
            or self.noise_multiplier <= 0
            or not 0 <= self.max_noise < 1
        ):
            raise CrossoverRuntimeError("resident speed policy is unsupported")
        for field in ("calibration_digest", "calibration_context_digest"):
            try:
                require_sha256_hex(getattr(self, field), field=field)
            except ValueError as exc:
                raise CrossoverRuntimeError(str(exc)) from None

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.qualification.resident-speed-policy",
            {
                "borderline_band": "one_min_margin_around_required",
                **self.to_dict(),
                "read_order": ["B", "C", "B_prime", "C_prime", "B_double_prime"],
                "timing": "serialized_resident_host_time",
            },
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "calibration_context_digest": self.calibration_context_digest,
            "calibration_digest": self.calibration_digest,
            "max_noise": format(self.max_noise, ".17g"),
            "max_qualification_seconds": self.max_qualification_seconds,
            "max_stage_seconds": self.max_stage_seconds,
            "min_margin": format(self.min_margin, ".17g"),
            "noise_multiplier": format(self.noise_multiplier, ".17g"),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ResidentSpeedPolicy":
        fields = {
            "calibration_context_digest",
            "calibration_digest",
            "max_noise",
            "max_qualification_seconds",
            "max_stage_seconds",
            "min_margin",
            "noise_multiplier",
            "version",
        }
        if type(value) is not dict or set(value) != fields:
            raise CrossoverRuntimeError("resident speed policy fields differ")
        try:
            result = cls(
                max_stage_seconds=value["max_stage_seconds"],  # type: ignore[arg-type]
                min_margin=float(value["min_margin"]),
                noise_multiplier=float(value["noise_multiplier"]),
                max_noise=float(value["max_noise"]),
                calibration_digest=value["calibration_digest"],  # type: ignore[arg-type]
                calibration_context_digest=value["calibration_context_digest"],  # type: ignore[arg-type]
                version=value["version"],  # type: ignore[arg-type]
                max_qualification_seconds=value["max_qualification_seconds"],  # type: ignore[arg-type]
            )
            if result.to_dict() != value:
                raise CrossoverRuntimeError("resident speed policy is noncanonical")
            return result
        except (TypeError, ValueError) as exc:
            raise CrossoverRuntimeError("resident speed policy is malformed") from exc

    @classmethod
    def from_calibration(
        cls,
        *,
        max_stage_seconds: int,
        max_qualification_seconds: int = 7_200,
        calibration: object,
        context: object,
    ) -> "ResidentSpeedPolicy":
        from optima.eval.calibration import (
            CalibrationContext,
            CalibrationManifest,
            decimal_value,
        )

        if (
            type(calibration) is not CalibrationManifest
            or type(context) is not CalibrationContext
            or not calibration.thresholds_frozen
        ):
            raise CrossoverRuntimeError("resident speed calibration is not frozen")
        try:
            calibration.require_context(context)
        except ValueError as exc:
            raise CrossoverRuntimeError(str(exc)) from None
        return cls(
            max_stage_seconds=max_stage_seconds,
            min_margin=float(decimal_value(calibration.speed.min_margin)),
            noise_multiplier=float(decimal_value(calibration.speed.noise_multiplier)),
            max_noise=float(decimal_value(calibration.speed.max_noise)),
            calibration_digest=calibration.digest,
            calibration_context_digest=context.digest,
            max_qualification_seconds=max_qualification_seconds,
        )


@dataclass(frozen=True)
class ResidentArmPlan:
    launch: EngineLaunchSpec
    binding: TrustedLaunchBinding
    session_plan: SessionExecutionPlan
    executor_namespace_digest: str
    runtime_resource_policy_digest: str
    device_configuration_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.launch) is not EngineLaunchSpec
            or type(self.binding) is not TrustedLaunchBinding
            or type(self.session_plan) is not SessionExecutionPlan
            or self.session_plan.launch_digest != self.launch.digest
            or self.session_plan.expected_engine_config_digest
            != self.launch.engine_config_digest
            or self.session_plan.audit_policy is not None
            or self.session_plan.engine_config.disable_cuda_graph
        ):
            raise CrossoverRuntimeError(
                "resident speed arm must be one exact graph-on, audit-free launch"
            )
        for field in (
            "executor_namespace_digest",
            "runtime_resource_policy_digest",
            "device_configuration_digest",
        ):
            try:
                require_sha256_hex(getattr(self, field), field=field)
            except ValueError as exc:
                raise CrossoverRuntimeError(str(exc)) from None


def _workload(plan: SessionExecutionPlan) -> tuple[object, ...]:
    return (
        plan.engine_config,
        plan.prompt_batches,
        plan.warmup_count,
        plan.conditioning_count,
        plan.max_new_tokens,
        plan.top_logprobs_num,
        plan.temperature,
    )


@dataclass(frozen=True)
class ResidentCrossoverPlan:
    selected_delta_digest: str
    baseline: ResidentArmPlan
    candidate: ResidentArmPlan
    policy: ResidentSpeedPolicy

    def __post_init__(self) -> None:
        try:
            selected = require_sha256_hex(
                self.selected_delta_digest, field="selected_delta_digest"
            )
        except ValueError as exc:
            raise CrossoverRuntimeError(str(exc)) from None
        object.__setattr__(self, "selected_delta_digest", selected)
        if (
            type(self.baseline) is not ResidentArmPlan
            or type(self.candidate) is not ResidentArmPlan
            or type(self.policy) is not ResidentSpeedPolicy
            or _workload(self.baseline.session_plan)
            != _workload(self.candidate.session_plan)
        ):
            raise CrossoverRuntimeError("resident crossover plan is inconsistent")
        allowed_differences = {
            "stack_digest",
            "tree_digest",
            "native_build_spec_digest",
            "resource_policy_digest",
        }
        common = set(self.baseline.launch.__dataclass_fields__) - {
            "hardware",
            *allowed_differences,
        }
        if any(
            getattr(self.baseline.launch, field)
            != getattr(self.candidate.launch, field)
            for field in common
        ):
            raise CrossoverRuntimeError("resident arms differ outside contribution identity")
        left, right = self.baseline.launch.hardware, self.candidate.launch.hardware
        shape = lambda row: (
            row.visible_gpu_count,
            row.architecture,
            row.topology_class,
            row.tp_size,
            row.ep_size,
            row.dp_size,
        )
        if shape(left) != shape(right) or left.visible_gpu_count != left.tp_size:
            raise CrossoverRuntimeError("resident TP lanes are not equivalent")
        if set(self.baseline.binding.physical_hardware.physical_gpu_ids) & set(
            self.candidate.binding.physical_hardware.physical_gpu_ids
        ):
            raise CrossoverRuntimeError("resident TP lanes overlap physical GPUs")
        if (
            self.baseline.executor_namespace_digest
            == self.candidate.executor_namespace_digest
        ):
            raise CrossoverRuntimeError(
                "resident TP lanes require distinct executor namespaces"
            )

    @property
    def digest(self) -> str:
        def arm(value: ResidentArmPlan) -> dict[str, object]:
            physical = value.binding.physical_hardware
            return {
                "device_policy": physical.device_policy_digest,
                "device_configuration": value.device_configuration_digest,
                "executor_namespace": value.executor_namespace_digest,
                "launch": value.launch.digest,
                "physical_gpu_ids": list(physical.physical_gpu_ids),
                "runtime_resource_policy": value.runtime_resource_policy_digest,
                "session_workload": marginal_workload_digest(value.session_plan),
                "topology": physical.topology_digest,
            }

        return canonical_digest(
            "optima.qualification.resident-crossover-plan",
            {
                "baseline": arm(self.baseline),
                "candidate": arm(self.candidate),
                "policy": self.policy.digest,
                "selected_delta": self.selected_delta_digest,
            },
        )

    @property
    def baseline_lane_digest(self) -> str:
        return _expected_lane_digest(self.baseline)

    @property
    def candidate_lane_digest(self) -> str:
        return _expected_lane_digest(self.candidate)


def _expanded(plan: SessionExecutionPlan, reads: int) -> SessionExecutionPlan:
    # Repeat the complete read, including its validator-owned warmup.  The model
    # remains loaded; only the cheap workload conditioning repeats between arms.
    return replace(plan, prompt_batches=plan.prompt_batches * reads)


def _expected_lane_digest(arm: ResidentArmPlan) -> str:
    physical = arm.binding.physical_hardware
    return canonical_digest(
        "optima.qualification.resident-lane",
        {
            "configuration": arm.device_configuration_digest,
            "namespace": arm.executor_namespace_digest,
            "physical_gpu_ids": list(physical.physical_gpu_ids),
            "policy": physical.device_policy_digest,
            "launch_resource_policy": arm.launch.resource_policy_digest,
            "runtime_policy": arm.runtime_resource_policy_digest,
            "topology": arm.launch.hardware.topology_digest,
        },
    )


def _lane_digest(executor: OCIEngineExecutor, arm: ResidentArmPlan) -> str:
    policy = executor.device_policy
    physical = arm.binding.physical_hardware
    if (
        policy.policy_sha256 != arm.launch.hardware.device_policy_digest
        or physical.device_policy_digest != policy.policy_sha256
        or tuple(map(str, policy.physical_gpu_ids)) != physical.physical_gpu_ids
        or executor.manager.namespace_digest != arm.executor_namespace_digest
        or executor.config.prebuild.policy.resource_policy_digest
        != arm.launch.resource_policy_digest
        or executor.config.runtime.digest != arm.runtime_resource_policy_digest
        or policy.configuration_sha256 != arm.device_configuration_digest
    ):
        raise CrossoverRuntimeError("executor and resident lane binding differ")
    return _expected_lane_digest(arm)


@dataclass(frozen=True)
class ResidentReadRate:
    role: str
    lane_digest: str
    launch_digest: str
    session_id: str
    first_batch_index: int
    last_batch_index: int
    first_timed_batch_index: int
    last_timed_batch_index: int
    conditioning_tokens: int
    timed_tokens: int
    charged_tokens: int
    conditioning_seconds: float
    timed_seconds: float
    charged_seconds: float
    tokens_per_second: float

    def __post_init__(self) -> None:
        if (
            self.role not in {"B", "C", "B_prime", "C_prime", "B_double_prime"}
            or not isinstance(self.session_id, str)
            or len(self.session_id) != 32
            or any(char not in "0123456789abcdef" for char in self.session_id)
            or self.session_id == "0" * 32
            or any(
                type(value) is not int
                for value in (
                    self.first_batch_index,
                    self.last_batch_index,
                    self.first_timed_batch_index,
                    self.last_timed_batch_index,
                )
            )
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.conditioning_tokens,
                    self.timed_tokens,
                    self.charged_tokens,
                )
            )
            or self.charged_tokens
            != self.conditioning_tokens + self.timed_tokens
            or not (
                0 <= self.first_batch_index
                <= self.first_timed_batch_index
                <= self.last_timed_batch_index
                <= self.last_batch_index
            )
            or any(
                not math.isfinite(value) or value <= 0
                for value in (
                    self.conditioning_seconds,
                    self.timed_seconds,
                    self.charged_seconds,
                )
            )
            or self.charged_seconds
            != self.conditioning_seconds + self.timed_seconds
            or self.tokens_per_second
            != self.charged_tokens / self.charged_seconds
        ):
            raise CrossoverRuntimeError("resident read rate is malformed")
        for field in ("lane_digest", "launch_digest"):
            try:
                require_sha256_hex(getattr(self, field), field=field)
            except ValueError as exc:
                raise CrossoverRuntimeError(str(exc)) from None

    def to_dict(self) -> dict[str, object]:
        return {
            "batches": [self.first_batch_index, self.last_batch_index],
            "charged_seconds": format(self.charged_seconds, ".17g"),
            "charged_tokens": self.charged_tokens,
            "conditioning_seconds": format(self.conditioning_seconds, ".17g"),
            "conditioning_tokens": self.conditioning_tokens,
            "lane_digest": self.lane_digest,
            "launch_digest": self.launch_digest,
            "role": self.role,
            "session_id": self.session_id,
            "timed_batches": [
                self.first_timed_batch_index,
                self.last_timed_batch_index,
            ],
            "timed_seconds": format(self.timed_seconds, ".17g"),
            "timed_tokens": self.timed_tokens,
        }

    @classmethod
    def from_dict(cls, value: object) -> "ResidentReadRate":
        fields = {
            "batches",
            "charged_seconds",
            "charged_tokens",
            "conditioning_seconds",
            "conditioning_tokens",
            "lane_digest",
            "launch_digest",
            "role",
            "session_id",
            "timed_batches",
            "timed_seconds",
            "timed_tokens",
        }
        if (
            type(value) is not dict
            or set(value) != fields
            or type(value["batches"]) is not list
            or len(value["batches"]) != 2
            or type(value["timed_batches"]) is not list
            or len(value["timed_batches"]) != 2
        ):
            raise CrossoverRuntimeError("resident rate fields differ")
        try:
            conditioning_seconds = float(value["conditioning_seconds"])
            timed_seconds = float(value["timed_seconds"])
            charged_seconds = float(value["charged_seconds"])
            conditioning_tokens = value["conditioning_tokens"]
            timed_tokens = value["timed_tokens"]
            charged_tokens = value["charged_tokens"]
            result = cls(
                value["role"],  # type: ignore[arg-type]
                value["lane_digest"],  # type: ignore[arg-type]
                value["launch_digest"],  # type: ignore[arg-type]
                value["session_id"],  # type: ignore[arg-type]
                value["batches"][0],  # type: ignore[index,arg-type]
                value["batches"][1],  # type: ignore[index,arg-type]
                value["timed_batches"][0],  # type: ignore[index,arg-type]
                value["timed_batches"][1],  # type: ignore[index,arg-type]
                conditioning_tokens,  # type: ignore[arg-type]
                timed_tokens,  # type: ignore[arg-type]
                charged_tokens,  # type: ignore[arg-type]
                conditioning_seconds,
                timed_seconds,
                charged_seconds,
                charged_tokens / charged_seconds,  # type: ignore[operator]
            )
            if result.to_dict() != value:
                raise CrossoverRuntimeError("resident rate is noncanonical")
            return result
        except (TypeError, ValueError, ZeroDivisionError) as exc:
            raise CrossoverRuntimeError("resident rate is malformed") from exc


def _rate(
    role: str,
    lane_digest: str,
    controller: OpenedOuterSession,
    template: SessionExecutionPlan,
) -> ResidentReadRate:
    first = controller.next_batch_index
    rows = tuple(
        controller.execute_next() for _ in range(len(template.prompt_batches))
    )
    timed = rows[template.warmup_count :]
    conditioning_start = template.warmup_count - template.conditioning_count
    conditioning = rows[conditioning_start : template.warmup_count]
    if (
        not rows
        or any(type(row) is not BatchExecutionEvidence or row.audit_receipts for row in rows)
        or tuple(row.batch_index for row in rows)
        != tuple(range(rows[0].batch_index, rows[-1].batch_index + 1))
        or tuple(
            controller.plan.prompt_batches[row.batch_index] for row in rows
        )
        != template.prompt_batches
        or not timed
        or not conditioning
    ):
        raise CrossoverRuntimeError("resident read batches are incomplete")
    conditioning_seconds = (
        timed[0].request_started_at - conditioning[0].request_started_at
    )
    timed_seconds = (
        timed[-1].response_completed_at - timed[0].request_started_at
    )
    conditioning_tokens = sum(row.token_numerator for row in conditioning)
    timed_tokens = sum(row.token_numerator for row in timed)
    charged_seconds = conditioning_seconds + timed_seconds
    charged_tokens = conditioning_tokens + timed_tokens
    return ResidentReadRate(
        role,
        lane_digest,
        controller.plan.launch_digest,
        controller.session_id,
        first,
        first + len(rows) - 1,
        timed[0].batch_index,
        timed[-1].batch_index,
        conditioning_tokens,
        timed_tokens,
        charged_tokens,
        float(conditioning_seconds),
        float(timed_seconds),
        float(charged_seconds),
        float(charged_tokens / charged_seconds),
    )


class _Schedule:
    def __init__(self) -> None:
        self.condition = threading.Condition()
        self.values: dict[str, object] = {}
        self.failure: BaseException | None = None

    def put(self, key: str, value: object = True) -> None:
        with self.condition:
            if key in self.values:
                raise CrossoverRuntimeError(f"resident schedule repeated {key}")
            self.values[key] = value
            self.condition.notify_all()

    def fail(self, exc: BaseException) -> None:
        with self.condition:
            if self.failure is None:
                self.failure = exc
            self.condition.notify_all()

    def get(
        self, key: str, *, deadline: float, clock: Callable[[], float]
    ) -> object:
        with self.condition:
            while key not in self.values:
                if self.failure is not None:
                    raise CrossoverRuntimeError(
                        f"resident peer failed: {self.failure}"
                    ) from self.failure
                remaining = deadline - float(clock())
                if not math.isfinite(remaining) or remaining <= 0:
                    raise CrossoverRuntimeError(
                        "resident speed stage exceeded its deadline"
                    )
                self.condition.wait(timeout=min(0.1, remaining))
            return self.values[key]


def _disposition(
    verdict: SpeedupVerdict, margin: float
) -> SpeedStageDecision | None:
    if not verdict.confident:
        return None
    if verdict.speedup <= verdict.required - margin:
        return SpeedStageDecision.FAIL
    if verdict.speedup >= verdict.required + margin:
        return SpeedStageDecision.PASS
    return None


def _final(verdict: SpeedupVerdict) -> SpeedStageDecision:
    if not verdict.confident:
        return SpeedStageDecision.NO_DECISION
    return SpeedStageDecision.PASS if verdict.passed_speedup else SpeedStageDecision.FAIL


def _execution_digest(value: EngineExecutionEvidence) -> str:
    return canonical_digest(
        "optima.qualification.resident-engine-execution",
        {
            "argv": value.runtime_argv_sha256,
            "batches": [
                [
                    row.batch_index,
                    row.request_id,
                    row.nonce,
                    format(row.request_started_at, ".17g"),
                    format(row.response_completed_at, ".17g"),
                    row.token_numerator,
                ]
                for row in value.session.batches
            ],
            "devices": [
                [row.launch_id, row.sequence, list(row.selected_physical_gpu_ids)]
                for row in value.device_receipts
            ],
            "launch": value.launch_digest,
            "native": value.native_publication_digest,
            "resource_policy": value.resource_policy_digest,
            "schema": value.schema,
            "session": value.session.session_id,
        },
    )


def _validate_resident_execution(
    execution: EngineExecutionEvidence,
    arm: ResidentArmPlan,
    *,
    reads: int,
) -> None:
    plan = _expanded(arm.session_plan, reads)
    session = execution.session
    if (
        execution.schema != "optima.oci-resident-engine-execution.v1"
        or execution.launch_digest != arm.launch.digest
        or execution.resource_policy_digest != arm.runtime_resource_policy_digest
        or type(session) is not SessionExecutionEvidence
        or session.launch_digest != arm.launch.digest
        or session.preflight != plan.expected_preflight
        or session.warmup_count != plan.warmup_count
        or session.conditioning_count != plan.conditioning_count
        or len(session.batches) != len(plan.prompt_batches)
    ):
        raise CrossoverRuntimeError("resident execution differs from its sealed arm")
    previous = session.ready_completed_at
    request_ids: set[str] = set()
    nonces: set[str] = set()
    for index, (row, prompts) in enumerate(
        zip(session.batches, plan.prompt_batches, strict=True)
    ):
        tokens = len(prompts) * plan.max_new_tokens
        if (
            type(row) is not BatchExecutionEvidence
            or row.batch_index != index
            or row.request_id in request_ids
            or row.nonce in nonces
            or row.request_started_at < previous
            or row.response_completed_at <= row.request_started_at
            or row.token_numerator != tokens
            or row.evidence.observed_tokens != tokens
            or row.audit_receipts
        ):
            raise CrossoverRuntimeError("resident execution batch evidence is malformed")
        request_ids.add(row.request_id)
        nonces.add(row.nonce)
        previous = row.response_completed_at
    if session.session_completed_at < previous:
        raise CrossoverRuntimeError("resident execution cleanup predates its last batch")
    receipts = execution.device_receipts
    physical = arm.binding.physical_hardware.physical_gpu_ids
    if type(receipts) is not tuple or len(receipts) != 3:
        raise CrossoverRuntimeError("resident execution lacks device receipt coverage")
    if all(value.isdecimal() for value in physical) and any(
        tuple(row.selected_physical_gpu_ids) != tuple(map(int, physical))
        for row in receipts
    ):
        raise CrossoverRuntimeError("resident execution ran on another physical lane")
    if any(
        getattr(row, "policy_sha256", arm.launch.hardware.device_policy_digest)
        != arm.launch.hardware.device_policy_digest
        or getattr(row, "configuration_sha256", arm.device_configuration_digest)
        != arm.device_configuration_digest
        for row in receipts
    ):
        raise CrossoverRuntimeError("resident execution changed device authority")


def _recomputed_rate(
    rate: ResidentReadRate,
    execution: EngineExecutionEvidence,
    arm: ResidentArmPlan,
) -> ResidentReadRate:
    plan = arm.session_plan
    start = rate.first_batch_index
    stop = start + len(plan.prompt_batches)
    if stop > len(execution.session.batches):
        raise CrossoverRuntimeError("resident rate exceeds its retained session")
    rows = execution.session.batches[start:stop]
    timed = rows[plan.warmup_count :]
    conditioning_start = plan.warmup_count - plan.conditioning_count
    conditioning = rows[conditioning_start : plan.warmup_count]
    if (
        not timed
        or not conditioning
        or rate.last_batch_index != stop - 1
        or tuple(
            _expanded(plan, stop // len(plan.prompt_batches)).prompt_batches[
                row.batch_index
            ]
            for row in rows
        )
        != plan.prompt_batches
    ):
        raise CrossoverRuntimeError("resident rate does not name one complete read")
    conditioning_seconds = (
        timed[0].request_started_at - conditioning[0].request_started_at
    )
    timed_seconds = (
        timed[-1].response_completed_at - timed[0].request_started_at
    )
    conditioning_tokens = sum(row.token_numerator for row in conditioning)
    timed_tokens = sum(row.token_numerator for row in timed)
    charged_seconds = conditioning_seconds + timed_seconds
    charged_tokens = conditioning_tokens + timed_tokens
    return ResidentReadRate(
        rate.role,
        _expected_lane_digest(arm),
        arm.launch.digest,
        execution.session.session_id,
        start,
        stop - 1,
        timed[0].batch_index,
        timed[-1].batch_index,
        conditioning_tokens,
        timed_tokens,
        charged_tokens,
        float(conditioning_seconds),
        float(timed_seconds),
        float(charged_seconds),
        float(charged_tokens / charged_seconds),
    )


@dataclass(frozen=True)
class ResidentCrossoverEvidence:
    plan_digest: str
    selected_delta_digest: str
    policy: ResidentSpeedPolicy
    workload_digest: str
    baseline_lane_digest: str
    candidate_lane_digest: str
    baseline_execution: EngineExecutionEvidence
    candidate_execution: EngineExecutionEvidence
    baseline_quiescence: OCIQuiescenceReceipt
    candidate_quiescence: OCIQuiescenceReceipt
    rates: tuple[ResidentReadRate, ...]
    initial_verdict: SpeedupVerdict
    final_verdict: SpeedupVerdict
    escalated: bool
    decision: SpeedStageDecision
    exit_reason: str
    started_monotonic_s: float
    completed_monotonic_s: float

    def __post_init__(self) -> None:
        roles = (
            ("B", "C", "B_prime", "C_prime", "B_double_prime")
            if self.escalated
            else ("B", "C", "B_prime")
        )
        for field in (
            "plan_digest",
            "selected_delta_digest",
            "workload_digest",
            "baseline_lane_digest",
            "candidate_lane_digest",
        ):
            try:
                require_sha256_hex(getattr(self, field), field=field)
            except ValueError as exc:
                raise CrossoverRuntimeError(str(exc)) from None
        if (
            type(self.policy) is not ResidentSpeedPolicy
            or type(self.baseline_execution) is not EngineExecutionEvidence
            or type(self.candidate_execution) is not EngineExecutionEvidence
            or type(self.baseline_quiescence) is not OCIQuiescenceReceipt
            or type(self.candidate_quiescence) is not OCIQuiescenceReceipt
            or type(self.initial_verdict) is not SpeedupVerdict
            or type(self.final_verdict) is not SpeedupVerdict
            or type(self.escalated) is not bool
            or type(self.decision) is not SpeedStageDecision
            or tuple(row.role for row in self.rates) != roles
            or self.baseline_lane_digest == self.candidate_lane_digest
            or self.exit_reason
            != ("borderline_" if self.escalated else "clear_")
            + self.decision.value.lower()
            or not all(
                math.isfinite(value)
                for value in (
                    self.started_monotonic_s,
                    self.completed_monotonic_s,
                )
            )
            or self.completed_monotonic_s <= self.started_monotonic_s
            or self.completed_monotonic_s - self.started_monotonic_s
            > self.policy.max_stage_seconds
        ):
            raise CrossoverRuntimeError("resident crossover evidence is malformed")

    def regrade(self, plan: ResidentCrossoverPlan) -> SpeedupVerdict:
        """Recompute the adaptive grade only from the sealed plan and raw spans."""

        if (
            type(plan) is not ResidentCrossoverPlan
            or self.plan_digest != plan.digest
            or self.selected_delta_digest != plan.selected_delta_digest
            or self.policy != plan.policy
            or self.workload_digest
            != marginal_workload_digest(plan.baseline.session_plan)
            or self.baseline_lane_digest != _expected_lane_digest(plan.baseline)
            or self.candidate_lane_digest != _expected_lane_digest(plan.candidate)
            or self.baseline_quiescence.namespace_digest
            != plan.baseline.executor_namespace_digest
            or self.candidate_quiescence.namespace_digest
            != plan.candidate.executor_namespace_digest
            or any(
                (row.container_ids, row.lease_records, row.resource_entries)
                != ((), (), ())
                for row in (
                    self.baseline_quiescence,
                    self.candidate_quiescence,
                )
            )
        ):
            raise CrossoverRuntimeError("resident evidence names another sealed plan")
        baseline_rates = tuple(
            row for row in self.rates if row.role.startswith("B")
        )
        candidate_rates = tuple(
            row for row in self.rates if row.role.startswith("C")
        )
        expected_baseline_reads = 3 if self.escalated else 2
        expected_candidate_reads = 2 if self.escalated else 1
        _validate_resident_execution(
            self.baseline_execution,
            plan.baseline,
            reads=expected_baseline_reads,
        )
        _validate_resident_execution(
            self.candidate_execution,
            plan.candidate,
            reads=expected_candidate_reads,
        )
        for rows, execution, arm in (
            (baseline_rates, self.baseline_execution, plan.baseline),
            (candidate_rates, self.candidate_execution, plan.candidate),
        ):
            block = len(arm.session_plan.prompt_batches)
            if tuple(row.first_batch_index for row in rows) != tuple(
                range(0, block * len(rows), block)
            ) or any(_recomputed_rate(row, execution, arm) != row for row in rows):
                raise CrossoverRuntimeError("resident rate spans do not independently regrade")
        initial = score_speedup(
            [baseline_rates[0].tokens_per_second, baseline_rates[1].tokens_per_second],
            [candidate_rates[0].tokens_per_second],
            min_margin=plan.policy.min_margin,
            k=plan.policy.noise_multiplier,
            max_noise=plan.policy.max_noise,
        )
        disposition = _disposition(initial, plan.policy.min_margin)
        if disposition is None:
            if not self.escalated:
                raise CrossoverRuntimeError("borderline resident evidence omitted repeat reads")
            final = score_speedup(
                [row.tokens_per_second for row in baseline_rates],
                [row.tokens_per_second for row in candidate_rates],
                min_margin=plan.policy.min_margin,
                k=plan.policy.noise_multiplier,
                max_noise=plan.policy.max_noise,
            )
            decision = _final(final)
        else:
            if self.escalated:
                raise CrossoverRuntimeError("clear resident evidence added unsealed reads")
            final, decision = initial, disposition
        if (
            self.initial_verdict != initial
            or self.final_verdict != final
            or self.decision is not decision
        ):
            raise CrossoverRuntimeError("resident speed headline does not regrade")
        return final

    @property
    def digest(self) -> str:
        verdict = lambda row: {
            "confident": row.confident,
            "noise": format(row.noise, ".17g"),
            "required": format(row.required, ".17g"),
            "speedup": format(row.speedup, ".17g"),
        }
        return canonical_digest(
            "optima.qualification.resident-crossover-speed",
            {
                "baseline_execution": _execution_digest(self.baseline_execution),
                "baseline_lane": self.baseline_lane_digest,
                "baseline_quiescence": self.baseline_quiescence.digest,
                "candidate_execution": _execution_digest(self.candidate_execution),
                "candidate_lane": self.candidate_lane_digest,
                "candidate_quiescence": self.candidate_quiescence.digest,
                "completed": format(self.completed_monotonic_s, ".17g"),
                "decision": self.decision.value,
                "escalated": self.escalated,
                "final": verdict(self.final_verdict),
                "initial": verdict(self.initial_verdict),
                "policy": self.policy.digest,
                "plan": self.plan_digest,
                "rates": [row.to_dict() for row in self.rates],
                "selected_delta": self.selected_delta_digest,
                "started": format(self.started_monotonic_s, ".17g"),
                "workload": self.workload_digest,
            },
        )


@dataclass(frozen=True)
class ResidentCandidateView:
    """Compatibility view of the singleton prepared candidate."""

    candidate: object
    execution: EngineExecutionEvidence

    @property
    def arm(self):
        return self.candidate.arm


@dataclass(frozen=True)
class ResidentMarginalLifecycleEvidence:
    """One singleton marginal lifecycle backed by exact resident role spans."""

    prepared: object
    plan: ResidentCrossoverPlan
    crossover: ResidentCrossoverEvidence
    quality_read: int = 1

    def __post_init__(self) -> None:
        from optima.eval.marginal_runtime import PreparedMarginalRuntime

        if (
            type(self.prepared) is not PreparedMarginalRuntime
            or len(self.prepared.candidates) != 1
            or type(self.plan) is not ResidentCrossoverPlan
            or type(self.crossover) is not ResidentCrossoverEvidence
            or type(self.quality_read) is not int
            or self.quality_read not in (1, 2)
        ):
            raise CrossoverRuntimeError("resident lifecycle is not a singleton authority")
        candidate = self.prepared.candidates[0]
        if (
            candidate.arm.selected_delta_digest != self.plan.selected_delta_digest
            or candidate.launch.digest != self.plan.candidate.launch.digest
            or candidate.session_plan != self.plan.candidate.session_plan
            or candidate.binding.launch_binding != self.plan.candidate.binding
            or self.prepared.baseline_launch.stack_digest
            != self.plan.baseline.launch.stack_digest
            or self.prepared.baseline_launch.tree_digest
            != self.plan.baseline.launch.tree_digest
            or (
                self.quality_read == 2
                and not self.crossover.escalated
            )
        ):
            raise CrossoverRuntimeError("resident lifecycle differs from its prepared arm")
        self.crossover.regrade(self.plan)

    @property
    def source(self):
        return self.prepared.source

    @property
    def candidates(self) -> tuple[ResidentCandidateView, ...]:
        return (
            ResidentCandidateView(
                self.prepared.candidates[0], self.crossover.candidate_execution
            ),
        )

    @property
    def candidates_repeat(self) -> tuple[ResidentCandidateView, ...]:
        return self.candidates if self.crossover.escalated else ()

    @property
    def final_baseline(self) -> EngineExecutionEvidence:
        """The resident baseline lifetime containing B-prime/B-double-prime."""

        return self.crossover.baseline_execution

    @property
    def timed_session_ids(self) -> frozenset[str]:
        return frozenset(
            {
                self.crossover.baseline_execution.session.session_id,
                self.crossover.candidate_execution.session.session_id,
            }
        )

    @property
    def role_names(self) -> tuple[str, str, str]:
        return (
            ("B", "C", "B_prime")
            if self.quality_read == 1
            else ("B_prime", "C_prime", "B_double_prime")
        )

    def role_batches(self, role: str) -> tuple[BatchExecutionEvidence, ...]:
        matches = tuple(row for row in self.crossover.rates if row.role == role)
        if len(matches) != 1 or role not in self.role_names:
            raise CrossoverRuntimeError("resident quality role is absent or ambiguous")
        rate = matches[0]
        execution = (
            self.crossover.baseline_execution
            if role.startswith("B")
            else self.crossover.candidate_execution
        )
        return execution.session.batches[
            rate.first_batch_index : rate.last_batch_index + 1
        ]

    def quality_leg(self, candidate_read: int) -> "ResidentMarginalLifecycleEvidence":
        return replace(self, quality_read=candidate_read)


def run_resident_crossover_speed(
    plan: ResidentCrossoverPlan,
    *,
    baseline_executor: OCIEngineExecutor,
    candidate_executor: OCIEngineExecutor,
    model_mount: TrustedArenaModelMountReceipt,
    deadline: float,
    clock: Callable[[], float] = time.monotonic,
) -> ResidentCrossoverEvidence:
    """Run the exact production/testnet speed scheduler for one candidate."""

    if (
        type(plan) is not ResidentCrossoverPlan
        or type(baseline_executor) is not OCIEngineExecutor
        or type(candidate_executor) is not OCIEngineExecutor
        or baseline_executor is candidate_executor
        or type(model_mount) is not TrustedArenaModelMountReceipt
    ):
        raise CrossoverRuntimeError("resident crossover authorities are not exact")
    started = float(clock())
    thresholds = (deadline, started)
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in thresholds
    ):
        raise CrossoverRuntimeError("resident crossover thresholds are invalid")
    stage_deadline = min(float(deadline), started + plan.policy.max_stage_seconds)
    if stage_deadline <= started:
        raise CrossoverRuntimeError("resident speed stage has no wall-clock budget")
    baseline_lane = _lane_digest(baseline_executor, plan.baseline)
    candidate_lane = _lane_digest(candidate_executor, plan.candidate)
    if baseline_lane == candidate_lane:
        raise CrossoverRuntimeError("resident executors reused one lane namespace")
    baseline_plan = _expanded(plan.baseline.session_plan, 3)
    candidate_plan = _expanded(plan.candidate.session_plan, 2)
    schedule = _Schedule()

    def score(baselines: list[float], candidates: list[float]) -> SpeedupVerdict:
        return score_speedup(
            baselines,
            candidates,
            min_margin=plan.policy.min_margin,
            k=plan.policy.noise_multiplier,
            max_noise=plan.policy.max_noise,
        )

    def baseline_driver(controller: OpenedOuterSession) -> SessionExecutionEvidence:
        try:
            schedule.put("baseline_ready")
            schedule.get("candidate_ready", deadline=stage_deadline, clock=clock)
            before = _rate(
                "B", baseline_lane, controller, plan.baseline.session_plan
            )
            schedule.put("B", before)
            candidate = schedule.get("C", deadline=stage_deadline, clock=clock)
            after = _rate(
                "B_prime", baseline_lane, controller, plan.baseline.session_plan
            )
            schedule.put("B_prime", after)
            initial = score(
                [before.tokens_per_second, after.tokens_per_second],
                [candidate.tokens_per_second],  # type: ignore[union-attr]
            )
            disposition = _disposition(initial, plan.policy.min_margin)
            schedule.put("initial", initial)
            schedule.put("escalate", disposition is None)
            if disposition is None:
                repeat = schedule.get("C_prime", deadline=stage_deadline, clock=clock)
                third = _rate(
                    "B_double_prime",
                    baseline_lane,
                    controller,
                    plan.baseline.session_plan,
                )
                final = score(
                    [
                        before.tokens_per_second,
                        after.tokens_per_second,
                        third.tokens_per_second,
                    ],
                    [
                        candidate.tokens_per_second,  # type: ignore[union-attr]
                        repeat.tokens_per_second,  # type: ignore[union-attr]
                    ],
                )
                disposition = _final(final)
                schedule.put("B_double_prime", third)
            else:
                final = initial
            schedule.put("final", final)
            schedule.put("decision", disposition)
            return controller.finish(require_all=False)
        except BaseException as exc:
            schedule.fail(exc)
            raise

    def candidate_driver(controller: OpenedOuterSession) -> SessionExecutionEvidence:
        try:
            schedule.put("candidate_ready")
            schedule.get("baseline_ready", deadline=stage_deadline, clock=clock)
            schedule.get("B", deadline=stage_deadline, clock=clock)
            schedule.put(
                "C",
                _rate("C", candidate_lane, controller, plan.candidate.session_plan),
            )
            escalated = schedule.get("escalate", deadline=stage_deadline, clock=clock)
            if escalated is True:
                schedule.put(
                    "C_prime",
                    _rate(
                        "C_prime",
                        candidate_lane,
                        controller,
                        plan.candidate.session_plan,
                    ),
                )
                # Keep the candidate resident and idle until B-double-prime is
                # complete.  Tearing its CUDA context down concurrently would
                # contaminate the final charged baseline read.
                schedule.get(
                    "B_double_prime", deadline=stage_deadline, clock=clock
                )
            return controller.finish(require_all=False)
        except BaseException as exc:
            schedule.fail(exc)
            raise

    def execute(executor, arm, expanded_plan, driver):
        try:
            return executor.execute_opened(
                arm.launch,
                arm.binding,
                model_mount,
                expanded_plan,
                deadline=stage_deadline,
                driver=driver,
            )
        except BaseException as exc:
            schedule.fail(exc)
            raise

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="optima-resident"
    ) as pool:
        futures = (
            pool.submit(
                execute,
                baseline_executor,
                plan.baseline,
                baseline_plan,
                baseline_driver,
            ),
            pool.submit(
                execute,
                candidate_executor,
                plan.candidate,
                candidate_plan,
                candidate_driver,
            ),
        )
        executions: list[EngineExecutionEvidence] = []
        errors: list[BaseException] = []
        for future in futures:
            try:
                executions.append(future.result())
            except BaseException as exc:
                errors.append(exc)
    if errors:
        raise schedule.failure or errors[0]
    if len(executions) != 2 or any(
        type(row) is not EngineExecutionEvidence for row in executions
    ):
        raise CrossoverRuntimeError("resident speed returned incomplete evidence")
    initial = schedule.values["initial"]
    final = schedule.values["final"]
    decision = schedule.values["decision"]
    escalated = schedule.values["escalate"] is True
    if (
        type(initial) is not SpeedupVerdict
        or type(final) is not SpeedupVerdict
        or type(decision) is not SpeedStageDecision
    ):
        raise CrossoverRuntimeError("resident speed grade is incomplete")
    roles = (
        ("B", "C", "B_prime", "C_prime", "B_double_prime")
        if escalated
        else ("B", "C", "B_prime")
    )
    rates = tuple(schedule.values[role] for role in roles)
    if any(type(row) is not ResidentReadRate for row in rates):
        raise CrossoverRuntimeError("resident speed rates are incomplete")
    baseline_quiescence = baseline_executor.prove_quiescent()
    candidate_quiescence = candidate_executor.prove_quiescent()
    completed = float(clock())
    evidence = ResidentCrossoverEvidence(
        plan.digest,
        plan.selected_delta_digest,
        plan.policy,
        marginal_workload_digest(plan.baseline.session_plan),
        baseline_lane,
        candidate_lane,
        executions[0],
        executions[1],
        baseline_quiescence,
        candidate_quiescence,
        rates,  # type: ignore[arg-type]
        initial,
        final,
        escalated,
        decision,
        ("borderline_" if escalated else "clear_") + decision.value.lower(),
        started,
        completed,
    )
    evidence.regrade(plan)
    return evidence


__all__ = [
    "CrossoverRuntimeError",
    "ResidentArmPlan",
    "ResidentCrossoverEvidence",
    "ResidentCrossoverPlan",
    "ResidentMarginalLifecycleEvidence",
    "ResidentReadRate",
    "ResidentSpeedPolicy",
    "SpeedStageDecision",
    "run_resident_crossover_speed",
]
