"""Causal B/C/B-prime to pristine-T qualification authority.

This module joins already-authenticated component evidence.  It does not load a
candidate, choose an incumbent, crown a winner, or mutate settlement state.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, ClassVar, Protocol

from optima.eval.calibration import (
    CalibrationContext, CalibrationManifest, CalibrationThresholdPolicy,
    decimal_value, reopen_calibration_evidence,
)
from optima.eval.engine_launch import EngineLaunchSpec, TrustedLaunchBinding
from optima.eval.evidence_store import EvidenceArtifactRef, publish_evidence, reopen_evidence
from optima.eval.device_state import DeviceStateReceipt
from optima.eval.marginal_runtime import (
    MarginalLifecycleEvidence, PreparedMarginalRuntime, run_marginal_lifecycle,
)
from optima.eval.oci_backend import (
    OCIEngineExecutor, PristineReferenceExecutionEvidence,
    TrustedArenaModelMountReceipt, runtime_identity_from_preflight,
)
from optima.eval.oci_process import OCIQuiescenceReceipt
from optima.eval.oci_reference_session import ReferenceSessionPlan
from optima.eval.oci_session_protocol import EngineSessionConfig, RuntimePreflightFacts
from optima.eval.qualification import (
    DiscoveryExecutionGrade, DiscoveryExecutionRequirement,
    DiscoveryQualificationProfile,
    GraphVerificationEvidenceRef, GraphVerificationGrade, GraphVerificationRequirement,
    QualificationDecision, QualificationProfile, SelectionCommitment,
    SelectionEntropyReceipt, SelectionReceipt, _selected_prompt_texts, _trajectory_rows,
    candidate_lifecycle_digest, cohort_trajectory_digest, derived_hidden_task_plan_digest,
    grade_discovery_execution, qualification_identity_digest,
    reopen_discovery_execution_binding, reopen_graph_verification,
    selected_trajectory_digest,
    selected_trajectory_projection_digest, validate_quality_binding,
)
from optima.eval.reference_protocol import (
    MAX_DERIVED_LOGPROBS, MAX_PROMPTS, MAX_SUPPORT_UNION, MAX_SUPPORT_WIDTH,
    MAX_TOKENS, ROLE_NAMES, ReferencePromptInput, ReferenceRequest,
    ReferenceRoleInput, request_sha256,
)
from optima.eval.reference_quality import (
    RAW_QUALITY_DOMAIN, RAW_QUALITY_SCHEMA, RawHiddenTaskResult,
    RawPromptQualityEvidence, RawRolloutEvidence, RawTokenEvidence,
    ReferenceQualityRawArtifact, ReferenceQualityRawBinding, ReferenceQualityVerdict,
    distribution_from_f32_logprobs, reopen_reference_quality_evidence,
    score_reference_quality, target_nll_from_f32,
)
from optima.eval.scoring import (
    ChargedExecutionRate, MarginalSpeedProjection, _projection_digest,
    marginal_workload_digest, project_marginal_speed, score_speedup,
)
from optima.stack_identity import canonical_digest, canonical_json_bytes, require_sha256_hex
from optima.stack_manifest import EvaluationStackManifest

class QualificationRunnerError(RuntimeError):
    """Infrastructure/authority failure; callers must treat it as NO_DECISION."""

    decision = QualificationDecision.NO_DECISION
    retryable = True

ATTEMPT_DOMAIN = "qualification.cohort-attempt"
ATTEMPT_SCHEMA = "optima.qualification.cohort-attempt.v1"
DISCOVERY_ATTEMPT_DOMAIN = "qualification.discovery-attempt"
DISCOVERY_ATTEMPT_SCHEMA = "optima.qualification.discovery-attempt.v1"

def _strict(value: object, fields: set[str], label: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != fields:
        raise QualificationRunnerError(f"{label} fields do not match the schema")
    return value

def _encode_record(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _encode_record(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_encode_record(row) for row in value]
    if type(value) is float:
        return format(value, ".17g")
    if value is None or type(value) in {str, bool, int}:
        return value
    raise QualificationRunnerError(f"attempt contains unsupported {type(value).__name__}")

def _record_dict(value: object) -> dict[str, object]:
    result = _encode_record(value)
    if type(result) is not dict:
        raise QualificationRunnerError("attempt record is not a dataclass")
    return result

def _canonical_payload(payload: bytes) -> dict[str, object]:
    def reject(_value: str) -> None:
        raise QualificationRunnerError("qualification artifact contains a JSON float")

    def pairs(rows: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in rows:
            if key in result:
                raise QualificationRunnerError("qualification artifact repeats a JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), parse_float=reject,
                           parse_constant=reject, object_pairs_hook=pairs)
        if type(value) is not dict or canonical_json_bytes(value) != payload:
            raise QualificationRunnerError("qualification artifact is not canonical JSON")
    except QualificationRunnerError:
        raise
    except (UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        raise QualificationRunnerError(f"qualification artifact is invalid: {exc}") from None
    return value

class EntropyProvider(Protocol):
    def __call__(self, commitment: SelectionCommitment,
                 teardown: OCIQuiescenceReceipt) -> SelectionEntropyReceipt: ...

class HiddenJudge(Protocol):
    binding: "HiddenJudgeBinding"

    def __call__(self, *, prompt_digest: str, output_ids: tuple[int, ...],
                 task_digests: tuple[str, ...]) -> "HiddenJudgeReceipt": ...

@dataclass(frozen=True)
class HiddenJudgeBinding:
    hidden_corpus_commitment: str
    hidden_judge_digest: str
    hidden_task_policy_digest: str

    def __post_init__(self) -> None:
        for field in ("hidden_corpus_commitment", "hidden_judge_digest", "hidden_task_policy_digest"):
            object.__setattr__(self, field, require_sha256_hex(getattr(self, field), field=field))

    @property
    def digest(self) -> str:
        return canonical_digest("optima.qualification.hidden-judge-binding", {
            "hidden_corpus_commitment": self.hidden_corpus_commitment,
            "hidden_judge_digest": self.hidden_judge_digest,
            "hidden_task_policy_digest": self.hidden_task_policy_digest,
        })

@dataclass(frozen=True)
class HiddenJudgeReceipt:
    binding_digest: str
    prompt_digest: str
    output_ids_digest: str
    task_digests: tuple[str, ...]
    passed: tuple[bool, ...]

    def __post_init__(self) -> None:
        for field in ("binding_digest", "prompt_digest", "output_ids_digest"):
            object.__setattr__(self, field, require_sha256_hex(getattr(self, field), field=field))
        tasks, results = tuple(self.task_digests), tuple(self.passed)
        if (
            tasks != tuple(sorted(set(tasks)))
            or any(require_sha256_hex(row, field="hidden task") != row for row in tasks)
            or len(results) != len(tasks)
            or any(type(row) is not bool for row in results)
        ):
            raise QualificationRunnerError("hidden judge receipt is malformed")
        object.__setattr__(self, "task_digests", tasks)
        object.__setattr__(self, "passed", results)

def hidden_judge_output_digest(prompt_digest: str, output_ids: tuple[int, ...]) -> str:
    """Bind one hidden-judge answer to the exact sealed rollout tokens."""

    prompt = require_sha256_hex(prompt_digest, field="hidden judge prompt")
    ids = tuple(output_ids)
    if any(type(token) is not int or token < 0 for token in ids):
        raise QualificationRunnerError("hidden judge output IDs are malformed")
    return canonical_digest(
        "optima.qualification.hidden-judge-output",
        {"output_ids": list(ids), "prompt_digest": prompt},
    )

@dataclass(frozen=True)
class CandidateQualificationAuthority:
    selected_delta_digest: str
    profile: QualificationProfile
    graph_requirement: GraphVerificationRequirement
    graph_artifact_ref: EvidenceArtifactRef
    graph_evidence_ref: GraphVerificationEvidenceRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_delta_digest",
                           require_sha256_hex(self.selected_delta_digest, field="selected delta"))
        if (
            type(self.profile) is not QualificationProfile
            or type(self.graph_requirement) is not GraphVerificationRequirement
            or type(self.graph_artifact_ref) is not EvidenceArtifactRef
            or type(self.graph_evidence_ref) is not GraphVerificationEvidenceRef
            or self.profile.graph_requirement_digest != self.graph_requirement.digest
            or self.graph_requirement.binding.selected_delta_digest
            != self.selected_delta_digest
        ):
            raise QualificationRunnerError("candidate graph/profile authority is mismatched")


@dataclass(frozen=True)
class DiscoveryCandidateQualificationAuthority:
    """Pre-execution authority for one non-catalog discovery arm."""

    selected_delta_digest: str
    profile: DiscoveryQualificationProfile
    execution_requirement: DiscoveryExecutionRequirement

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_delta_digest",
            require_sha256_hex(self.selected_delta_digest, field="selected delta"),
        )
        if (
            type(self.profile) is not DiscoveryQualificationProfile
            or type(self.execution_requirement) is not DiscoveryExecutionRequirement
            or self.profile.execution_requirement_digest
            != self.execution_requirement.digest
            or self.execution_requirement.selected_delta_digest
            != self.selected_delta_digest
        ):
            raise QualificationRunnerError(
                "discovery execution/profile authority is mismatched"
            )


CandidateAuthority = (
    CandidateQualificationAuthority | DiscoveryCandidateQualificationAuthority
)

@dataclass(frozen=True)
class CausalQualificationInput:
    prepared: PreparedMarginalRuntime
    model_mount: TrustedArenaModelMountReceipt
    candidates: tuple[CandidateAuthority, ...]
    commitment: SelectionCommitment
    selection_secret: bytes
    evidence_root: Path
    calibration_threshold_policy: CalibrationThresholdPolicy
    calibration_manifest: CalibrationManifest
    calibration_context: CalibrationContext
    calibration_artifact_ref: EvidenceArtifactRef
    pristine_stack: EvaluationStackManifest
    pristine_launch: EngineLaunchSpec
    pristine_binding: TrustedLaunchBinding
    reference_engine_config: EngineSessionConfig
    reference_preflight: RuntimePreflightFacts
    expected_launch_resource_policy_digest: str
    expected_runtime_resource_policy_digest: str
    expected_device_policy_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.prepared) is not PreparedMarginalRuntime
            or type(self.model_mount) is not TrustedArenaModelMountReceipt
            or type(self.commitment) is not SelectionCommitment
            or not isinstance(self.selection_secret, bytes)
            or len(self.selection_secret) < 32
            or type(self.calibration_threshold_policy) is not CalibrationThresholdPolicy
            or type(self.calibration_manifest) is not CalibrationManifest
            or type(self.calibration_context) is not CalibrationContext
            or type(self.calibration_artifact_ref) is not EvidenceArtifactRef
            or type(self.pristine_stack) is not EvaluationStackManifest
            or type(self.pristine_launch) is not EngineLaunchSpec
            or type(self.pristine_binding) is not TrustedLaunchBinding
            or type(self.reference_engine_config) is not EngineSessionConfig
            or type(self.reference_preflight) is not RuntimePreflightFacts
        ):
            raise QualificationRunnerError("causal qualification input is not exactly typed")
        root = Path(self.evidence_root)
        if not root.is_absolute() or root != Path(root.as_posix()):
            raise QualificationRunnerError("qualification evidence root must be canonical and absolute")
        object.__setattr__(self, "evidence_root", root)
        candidates = tuple(self.candidates)
        expected = tuple(row.arm.selected_delta_digest for row in self.prepared.candidates)
        from optima.discovery import DiscoveryArmPlan

        discovery = type(self.prepared.source) is DiscoveryArmPlan
        allowed = (
            {DiscoveryCandidateQualificationAuthority}
            if discovery
            else {CandidateQualificationAuthority}
        )
        if (
            not candidates
            or (discovery and len(candidates) != 1)
            or any(type(row) not in allowed for row in candidates)
            or tuple(row.selected_delta_digest for row in candidates) != expected
        ):
            raise QualificationRunnerError("candidate authority order differs from the sealed cohort")
        object.__setattr__(self, "candidates", candidates)
        for authority in candidates:
            profile = authority.profile
            support_union = min(
                MAX_SUPPORT_UNION,
                profile.tokens_per_prompt * profile.topk_width,
            )
            derived = (
                self.commitment.select_count
                * len(ROLE_NAMES)
                * profile.tokens_per_prompt
                * support_union
            )
            if (
                profile.tokens_per_prompt > MAX_TOKENS
                or profile.topk_width > MAX_SUPPORT_WIDTH
                or self.commitment.select_count > MAX_PROMPTS
                or derived > MAX_DERIVED_LOGPROBS
            ):
                raise QualificationRunnerError(
                    "sealed qualification can exceed the pristine-reference bounds"
                )
        for field in (
            "expected_launch_resource_policy_digest",
            "expected_runtime_resource_policy_digest",
            "expected_device_policy_digest",
        ):
            object.__setattr__(self, field, require_sha256_hex(getattr(self, field), field=field))


def _profile(authority: CandidateAuthority) -> QualificationProfile | DiscoveryQualificationProfile:
    return authority.profile


def _requirement(
    authority: CandidateAuthority,
) -> GraphVerificationRequirement | DiscoveryExecutionRequirement:
    if type(authority) is CandidateQualificationAuthority:
        return authority.graph_requirement
    if type(authority) is DiscoveryCandidateQualificationAuthority:
        return authority.execution_requirement
    raise QualificationRunnerError("candidate authority has an unsupported type")

def _decision(value: str | QualificationDecision) -> QualificationDecision:
    try:
        return value if type(value) is QualificationDecision else QualificationDecision(value)
    except (TypeError, ValueError) as exc:
        raise QualificationRunnerError("component decision is unsupported") from exc

def _aggregate_decision(graph: QualificationDecision, speed: QualificationDecision,
                        quality: QualificationDecision) -> QualificationDecision:
    values = (graph, speed, quality)
    if QualificationDecision.FAIL in values:
        return QualificationDecision.FAIL
    if QualificationDecision.NO_DECISION in values:
        return QualificationDecision.NO_DECISION
    return QualificationDecision.PASS

def _validated_rate(row: ChargedExecutionRate) -> ChargedExecutionRate:
    if type(row) is not ChargedExecutionRate:
        raise QualificationRunnerError("charged rate is not typed")
    require_sha256_hex(row.launch_digest, field="rate launch")
    counts = (row.conditioning_tokens, row.timed_tokens, row.charged_tokens)
    seconds = (row.conditioning_seconds, row.timed_seconds, row.charged_seconds)
    if (len(row.session_id) != 32 or any(char not in "0123456789abcdef" for char in row.session_id)
            or any(type(value) is not int or value <= 0 for value in counts)
            or counts[2] != counts[0] + counts[1]
            or any(type(value) is not float or not math.isfinite(value) or value <= 0 for value in seconds)
            or seconds[2] != seconds[0] + seconds[1]
            or row.tokens_per_second != counts[2] / seconds[2]):
        raise QualificationRunnerError("charged rate numerators or intervals are inconsistent")
    return row

def _rate_from_dict(value: object) -> ChargedExecutionRate:
    raw = dict(_strict(value, set(ChargedExecutionRate.__dataclass_fields__), "charged rate witness"))
    for field in ("conditioning_seconds", "timed_seconds", "charged_seconds", "tokens_per_second"):
        raw[field] = float(raw[field])
    return _validated_rate(ChargedExecutionRate(**raw))  # type: ignore[arg-type]

@dataclass(frozen=True)
class SpeedWitness:
    selected_delta_digest: str
    candidate_launch_digest: str
    calibration_digest: str
    calibration_context_digest: str
    workload_digest: str
    runtime_resource_policy_digest: str
    evidence_digest: str
    rates: tuple[ChargedExecutionRate, ChargedExecutionRate, ChargedExecutionRate]

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            if field != "rates":
                object.__setattr__(self, field, require_sha256_hex(getattr(self, field), field=field))
        rates = tuple(self.rates)
        if len(rates) != 3:
            raise QualificationRunnerError("speed witness must contain B/C/B-prime rates")
        rates = tuple(_validated_rate(row) for row in rates)
        object.__setattr__(self, "rates", rates)
        if _projection_digest(
            self.selected_delta_digest, self.candidate_launch_digest, self.calibration_digest,
            self.calibration_context_digest, self.workload_digest,
            self.runtime_resource_policy_digest, rates,
        ) != self.evidence_digest:
            raise QualificationRunnerError("speed witness evidence digest does not recompute")

    @classmethod
    def from_projection(cls, row: MarginalSpeedProjection, runtime_policy: str) -> "SpeedWitness":
        if type(row) is not MarginalSpeedProjection:
            raise QualificationRunnerError("speed projection is not typed")
        return cls(
            row.selected_delta_digest, row.candidate_launch_digest, row.calibration_digest,
            row.calibration_context_digest, row.workload_digest, runtime_policy,
            row.evidence_digest, (row.baseline_before, row.candidate, row.baseline_after),
        )

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: object) -> "SpeedWitness":
        raw = _strict(value, set(cls.__dataclass_fields__), "speed witness")
        return cls(**{**raw, "rates": tuple(_rate_from_dict(row) for row in raw["rates"])})  # type: ignore[arg-type]

    def regrade(self, calibration: CalibrationManifest, context: CalibrationContext) -> tuple[QualificationDecision, str]:
        if (
            self.calibration_digest != calibration.digest
            or self.calibration_context_digest != context.digest
            or self.workload_digest != context.workload_digest
            or not calibration.thresholds_frozen
        ):
            raise QualificationRunnerError("speed witness calibration authority differs")
        before, candidate, after = self.rates
        verdict = score_speedup(
            [before.tokens_per_second, after.tokens_per_second], candidate.tokens_per_second,
            min_margin=float(decimal_value(calibration.speed.min_margin)),
            k=float(decimal_value(calibration.speed.noise_multiplier)),
            max_noise=float(decimal_value(calibration.speed.max_noise)),
        )
        grade = (
            QualificationDecision.NO_DECISION
            if not verdict.confident
            else QualificationDecision.PASS
            if verdict.passed_speedup
            else QualificationDecision.FAIL
        )
        return grade, format(verdict.speedup, ".17g")

@dataclass(frozen=True)
class CandidateQualificationReport:
    _domain: ClassVar[str] = "optima.qualification.candidate-report.v1"
    selected_delta_digest: str
    marginal_arm_digest: str
    candidate_launch_digest: str
    target_id: str
    profile_digest: str
    calibration_digest: str
    graph_grade_digest: str
    graph_decision: QualificationDecision
    speed_evidence_digest: str
    speed_decision: QualificationDecision
    speedup: str
    quality_evidence_digest: str
    quality_decision: QualificationDecision
    candidate_mean_teacher_nll: str
    raw_quality_artifact: EvidenceArtifactRef
    raw_quality_binding: ReferenceQualityRawBinding
    speed_witness: SpeedWitness
    t_request_sha256: str
    decision: QualificationDecision
    reason: str
    retryable: bool

    def __post_init__(self) -> None:
        for field in (
            "selected_delta_digest", "marginal_arm_digest", "candidate_launch_digest",
            "profile_digest", "calibration_digest", "graph_grade_digest",
            "speed_evidence_digest", "quality_evidence_digest", "t_request_sha256",
        ):
            object.__setattr__(self, field, require_sha256_hex(getattr(self, field), field=field))
        for field in ("graph_decision", "speed_decision", "quality_decision", "decision"):
            object.__setattr__(self, field, _decision(getattr(self, field)))
        if (
            type(self.raw_quality_artifact) is not EvidenceArtifactRef
            or type(self.raw_quality_binding) is not ReferenceQualityRawBinding
            or type(self.speed_witness) is not SpeedWitness
        ):
            raise QualificationRunnerError("candidate evidence witness is not typed")
        expected = _aggregate_decision(
            self.graph_decision, self.speed_decision, self.quality_decision
        )
        if (
            self.decision is not expected
            or type(self.retryable) is not bool
            or self.retryable != (expected is QualificationDecision.NO_DECISION)
            or not isinstance(self.reason, str)
            or not self.reason
            or not isinstance(self.target_id, str)
            or not self.target_id
        ):
            raise QualificationRunnerError("candidate aggregate headline is inconsistent")
        for field in ("speedup", "candidate_mean_teacher_nll"):
            try:
                value = float(getattr(self, field))
            except (TypeError, ValueError) as exc:
                raise QualificationRunnerError(f"{field} is not numeric") from exc
            if not math.isfinite(value) or value < 0:
                raise QualificationRunnerError(f"{field} is nonfinite or negative")

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: object) -> "CandidateQualificationReport":
        raw = _strict(value, set(cls.__dataclass_fields__) - {"_domain"}, "candidate report")
        return cls(**{
            **raw,
            "raw_quality_artifact": EvidenceArtifactRef.from_dict(raw["raw_quality_artifact"]),
            "raw_quality_binding": ReferenceQualityRawBinding.from_dict(raw["raw_quality_binding"]),
            "speed_witness": SpeedWitness.from_dict(raw["speed_witness"]),
        })  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(self._domain, self.to_dict())


@dataclass(frozen=True)
class DiscoveryCandidateQualificationReport:
    _domain: ClassVar[str] = "optima.qualification.discovery-candidate-report.v1"
    selected_delta_digest: str
    discovery_arm_digest: str
    proposal_digest: str
    candidate_launch_digest: str
    profile_digest: str
    calibration_digest: str
    execution_grade: DiscoveryExecutionGrade
    speed_evidence_digest: str
    speed_decision: QualificationDecision
    speedup: str
    quality_evidence_digest: str
    quality_decision: QualificationDecision
    candidate_mean_teacher_nll: str
    raw_quality_artifact: EvidenceArtifactRef
    raw_quality_binding: ReferenceQualityRawBinding
    speed_witness: SpeedWitness
    t_request_sha256: str
    decision: QualificationDecision
    reason: str
    retryable: bool

    def __post_init__(self) -> None:
        for field in (
            "selected_delta_digest", "discovery_arm_digest", "proposal_digest",
            "candidate_launch_digest", "profile_digest", "calibration_digest",
            "speed_evidence_digest",
            "quality_evidence_digest", "t_request_sha256",
        ):
            object.__setattr__(
                self, field, require_sha256_hex(getattr(self, field), field=field)
            )
        for field in ("speed_decision", "quality_decision", "decision"):
            object.__setattr__(self, field, _decision(getattr(self, field)))
        if (
            type(self.execution_grade) is not DiscoveryExecutionGrade
            or type(self.raw_quality_artifact) is not EvidenceArtifactRef
            or type(self.raw_quality_binding) is not ReferenceQualityRawBinding
            or type(self.speed_witness) is not SpeedWitness
        ):
            raise QualificationRunnerError("discovery evidence witness is not typed")
        expected = _aggregate_decision(
            self.execution_grade.decision, self.speed_decision, self.quality_decision
        )
        if (
            self.decision is not expected
            or type(self.retryable) is not bool
            or self.retryable != (expected is QualificationDecision.NO_DECISION)
            or not isinstance(self.reason, str)
            or not self.reason
        ):
            raise QualificationRunnerError(
                "discovery candidate aggregate headline is inconsistent"
            )
        for field in ("speedup", "candidate_mean_teacher_nll"):
            try:
                value = float(getattr(self, field))
            except (TypeError, ValueError) as exc:
                raise QualificationRunnerError(f"{field} is not numeric") from exc
            if not math.isfinite(value) or value < 0:
                raise QualificationRunnerError(f"{field} is nonfinite or negative")

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryCandidateQualificationReport":
        raw = _strict(
            value,
            set(cls.__dataclass_fields__) - {"_domain"},
            "discovery candidate report",
        )
        return cls(**{
            **raw,
            "execution_grade": DiscoveryExecutionGrade.from_dict(
                raw["execution_grade"]
            ),
            "raw_quality_artifact": EvidenceArtifactRef.from_dict(
                raw["raw_quality_artifact"]
            ),
            "raw_quality_binding": ReferenceQualityRawBinding.from_dict(
                raw["raw_quality_binding"]
            ),
            "speed_witness": SpeedWitness.from_dict(raw["speed_witness"]),
        })  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(self._domain, self.to_dict())

def _device_witness(receipt: DeviceStateReceipt) -> tuple[object, ...]:
    if type(receipt) is not DeviceStateReceipt:
        raise QualificationRunnerError("T device receipt is not typed")
    return (
        receipt.schema, receipt.sequence, receipt.launch_id, receipt.phase,
        tuple(receipt.selected_physical_gpu_ids), receipt.configuration_sha256,
        receipt.policy_sha256, format(receipt.started_monotonic_s, ".17g"),
        format(receipt.completed_monotonic_s, ".17g"), receipt.consecutive_idle_samples,
        len(receipt.samples),
    )

def _validate_device_witness(row: object, phase: str) -> tuple[object, ...]:
    value = tuple(row) if isinstance(row, (list, tuple)) else ()
    if len(value) != 11 or value[0] != "optima.device-state-receipt.v1" or value[3] != phase:
        raise QualificationRunnerError("T device witness shape differs")
    if (type(value[1]) is not int or value[1] < 1 or not isinstance(value[2], str)
            or not isinstance(value[4], (list, tuple)) or not value[4]
            or any(type(gpu) is not int or gpu < 0 for gpu in value[4])
            or any(require_sha256_hex(value[index], field="T device digest") != value[index]
                   for index in (5, 6))
            or any(type(value[index]) is not int or value[index] < 1 for index in (9, 10))):
        raise QualificationRunnerError("T device witness identity differs")
    seconds = tuple(float(value[index]) for index in (7, 8))
    if any(not math.isfinite(item) or item < 0 for item in seconds) or seconds[1] < seconds[0]:
        raise QualificationRunnerError("T device witness time is invalid")
    return (*value[:4], tuple(value[4]), *value[5:])

@dataclass(frozen=True)
class ReferenceExecutionWitness:
    launch_digest: str
    runtime_identity_digest: str
    runtime_preflight_receipt_sha256: str
    arena_model_receipt_digest: str
    resource_policy_digest: str
    build_spec_digest: str
    native_publication_digest: str
    runtime_argv_sha256: str
    recovered_lease_ids: tuple[str, ...]
    session_digest: str
    session_id: str
    plan_digest: str
    request_plan_digest: str
    request_sha256: tuple[str, ...]
    evidence_frame_sha256: tuple[str, ...]
    device_receipts: tuple[tuple[object, ...], tuple[object, ...]]

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            if field not in {"request_sha256", "evidence_frame_sha256"} and (
                field.endswith("digest") or field.endswith("sha256")
            ):
                object.__setattr__(self, field, require_sha256_hex(getattr(self, field), field=field))
        leases, requests = tuple(self.recovered_lease_ids), tuple(self.request_sha256)
        evidence = tuple(self.evidence_frame_sha256)
        if len(set(leases)) != len(leases) or any(not isinstance(row, str) or not row for row in leases):
            raise QualificationRunnerError("T recovered leases are malformed")
        if (len(self.session_id) != 32 or any(char not in "0123456789abcdef" for char in self.session_id)
                or not requests or len(evidence) != len(requests)
                or any(require_sha256_hex(row, field="T frame") != row for row in requests + evidence)):
            raise QualificationRunnerError("T request coverage is malformed")
        receipts = tuple(_validate_device_witness(row, phase) for row, phase in
                         zip(self.device_receipts, ("pre", "post"), strict=True))
        object.__setattr__(self, "recovered_lease_ids", leases)
        object.__setattr__(self, "request_sha256", requests)
        object.__setattr__(self, "evidence_frame_sha256", evidence)
        object.__setattr__(self, "device_receipts", receipts)

    @classmethod
    def from_execution(cls, value: PristineReferenceExecutionEvidence, plan_digest: str) -> "ReferenceExecutionWitness":
        identity = value.runtime_identity
        return cls(
            value.launch_digest, canonical_digest("optima.qualification.reference-runtime", {
                "base": identity.base_engine_digest, "runtime": identity.runtime_digest,
                "validator_overlay": identity.validator_overlay_digest,
            }), value.runtime_preflight_receipt_sha256, value.arena_model_receipt_digest,
            value.resource_policy_digest, value.prebuild.build_spec_digest,
            value.native_publication_digest, value.runtime_argv_sha256,
            tuple(value.recovered_lease_ids), value.session.digest, value.session.session_id,
            plan_digest, value.session.request_plan_digest,
            tuple(row.request_sha256 for row in value.session.exchanges),
            tuple(row.evidence_frame_sha256 for row in value.session.exchanges),
            tuple(_device_witness(row) for row in value.device_receipts),
        )

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: object) -> "ReferenceExecutionWitness":
        raw = _strict(value, set(cls.__dataclass_fields__), "reference execution witness")
        return cls(**{**raw, "recovered_lease_ids": tuple(raw["recovered_lease_ids"]),
                      "request_sha256": tuple(raw["request_sha256"]),
                      "evidence_frame_sha256": tuple(raw["evidence_frame_sha256"]),
                      "device_receipts": tuple(tuple(row) for row in raw["device_receipts"])})  # type: ignore[arg-type]

def _quiescence_from_dict(value: object) -> OCIQuiescenceReceipt:
    raw = _strict(value, {
        "container_ids", "executor_id", "lease_records", "manager_instance_id",
        "namespace_digest", "observed_monotonic_s", "resource_entries", "schema", "sequence",
    }, "quiescence receipt")
    try:
        observed = float(raw["observed_monotonic_s"])
    except (TypeError, ValueError) as exc:
        raise QualificationRunnerError("quiescence timestamp is malformed") from exc
    return OCIQuiescenceReceipt(
        raw["schema"], raw["executor_id"], raw["manager_instance_id"],
        raw["namespace_digest"], raw["sequence"], observed,
        tuple(raw["lease_records"]), tuple(raw["resource_entries"]), tuple(raw["container_ids"]),
    )  # type: ignore[arg-type]

@dataclass(frozen=True)
class CohortQualificationAttempt:
    _domain: ClassVar[str] = "optima.qualification.cohort-attempt.v1"
    authority_digest: str
    source_digest: str
    cohort_trajectory_digest: str
    commitment: SelectionCommitment
    teardown_before_t: OCIQuiescenceReceipt
    entropy: SelectionEntropyReceipt
    entropy_observed_monotonic_s: float
    selection: SelectionReceipt
    reference_execution: ReferenceExecutionWitness
    teardown_after_t: OCIQuiescenceReceipt
    reports: tuple[CandidateQualificationReport, ...]

    def __post_init__(self) -> None:
        for field in (
            "authority_digest", "source_digest", "cohort_trajectory_digest",
        ):
            object.__setattr__(self, field, require_sha256_hex(getattr(self, field), field=field))
        if (
            type(self.commitment) is not SelectionCommitment
            or type(self.teardown_before_t) is not OCIQuiescenceReceipt
            or type(self.entropy) is not SelectionEntropyReceipt
            or type(self.selection) is not SelectionReceipt
            or type(self.reference_execution) is not ReferenceExecutionWitness
            or type(self.teardown_after_t) is not OCIQuiescenceReceipt
            or type(self.entropy_observed_monotonic_s) is not float
            or not math.isfinite(self.entropy_observed_monotonic_s)
        ):
            raise QualificationRunnerError("cohort attempt authority is not typed")
        reports = tuple(self.reports)
        if not reports or any(type(row) is not CandidateQualificationReport for row in reports):
            raise QualificationRunnerError("cohort reports are not typed")
        object.__setattr__(self, "reports", reports)
        before, after = self.teardown_before_t, self.teardown_after_t
        if (
            (before.executor_id, before.manager_instance_id, before.namespace_digest)
            != (after.executor_id, after.manager_instance_id, after.namespace_digest)
            or after.sequence != before.sequence + 1
            or not before.observed_monotonic_s <= self.entropy_observed_monotonic_s
            <= after.observed_monotonic_s
            or self.selection.commitment_digest != self.commitment.digest
            or self.selection.entropy_receipt_digest != self.entropy.digest
        ):
            raise QualificationRunnerError("cohort causal ordering or executor identity differs")

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: object) -> "CohortQualificationAttempt":
        raw = _strict(value, set(cls.__dataclass_fields__) - {"_domain"}, "cohort attempt")
        return cls(**{
            **raw,
            "commitment": SelectionCommitment.from_dict(raw["commitment"]),
            "entropy": SelectionEntropyReceipt.from_dict(raw["entropy"]),
            "entropy_observed_monotonic_s": float(raw["entropy_observed_monotonic_s"]),
            "selection": SelectionReceipt.from_dict(raw["selection"]),
            "reference_execution": ReferenceExecutionWitness.from_dict(raw["reference_execution"]),
            "teardown_before_t": _quiescence_from_dict(raw["teardown_before_t"]),
            "teardown_after_t": _quiescence_from_dict(raw["teardown_after_t"]),
            "reports": tuple(CandidateQualificationReport.from_dict(row) for row in raw["reports"]),
        })  # type: ignore[arg-type]

    @property
    def reference_plan_digest(self) -> str:
        return self.reference_execution.plan_digest

    @property
    def reference_session_digest(self) -> str:
        return self.reference_execution.session_digest

    @property
    def digest(self) -> str:
        return canonical_digest(self._domain, self.to_dict())


@dataclass(frozen=True)
class DiscoveryQualificationAttempt:
    _domain: ClassVar[str] = "optima.qualification.discovery-attempt.v1"
    authority_digest: str
    source_digest: str
    cohort_trajectory_digest: str
    commitment: SelectionCommitment
    teardown_before_t: OCIQuiescenceReceipt
    entropy: SelectionEntropyReceipt
    entropy_observed_monotonic_s: float
    selection: SelectionReceipt
    reference_execution: ReferenceExecutionWitness
    teardown_after_t: OCIQuiescenceReceipt
    reports: tuple[DiscoveryCandidateQualificationReport, ...]

    def __post_init__(self) -> None:
        for field in (
            "authority_digest", "source_digest", "cohort_trajectory_digest"
        ):
            object.__setattr__(
                self, field, require_sha256_hex(getattr(self, field), field=field)
            )
        if (
            type(self.commitment) is not SelectionCommitment
            or type(self.teardown_before_t) is not OCIQuiescenceReceipt
            or type(self.entropy) is not SelectionEntropyReceipt
            or type(self.selection) is not SelectionReceipt
            or type(self.reference_execution) is not ReferenceExecutionWitness
            or type(self.teardown_after_t) is not OCIQuiescenceReceipt
            or type(self.entropy_observed_monotonic_s) is not float
            or not math.isfinite(self.entropy_observed_monotonic_s)
        ):
            raise QualificationRunnerError(
                "discovery attempt authority is not typed"
            )
        reports = tuple(self.reports)
        if len(reports) != 1 or type(reports[0]) is not DiscoveryCandidateQualificationReport:
            raise QualificationRunnerError(
                "discovery attempt requires exactly one typed report"
            )
        object.__setattr__(self, "reports", reports)
        before, after = self.teardown_before_t, self.teardown_after_t
        if (
            (before.executor_id, before.manager_instance_id, before.namespace_digest)
            != (after.executor_id, after.manager_instance_id, after.namespace_digest)
            or after.sequence != before.sequence + 1
            or not before.observed_monotonic_s <= self.entropy_observed_monotonic_s
            <= after.observed_monotonic_s
            or self.selection.commitment_digest != self.commitment.digest
            or self.selection.entropy_receipt_digest != self.entropy.digest
        ):
            raise QualificationRunnerError(
                "discovery causal ordering or executor identity differs"
            )

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryQualificationAttempt":
        raw = _strict(
            value,
            set(cls.__dataclass_fields__) - {"_domain"},
            "discovery attempt",
        )
        return cls(**{
            **raw,
            "commitment": SelectionCommitment.from_dict(raw["commitment"]),
            "entropy": SelectionEntropyReceipt.from_dict(raw["entropy"]),
            "entropy_observed_monotonic_s": float(
                raw["entropy_observed_monotonic_s"]
            ),
            "selection": SelectionReceipt.from_dict(raw["selection"]),
            "reference_execution": ReferenceExecutionWitness.from_dict(
                raw["reference_execution"]
            ),
            "teardown_before_t": _quiescence_from_dict(raw["teardown_before_t"]),
            "teardown_after_t": _quiescence_from_dict(raw["teardown_after_t"]),
            "reports": tuple(
                DiscoveryCandidateQualificationReport.from_dict(row)
                for row in raw["reports"]
            ),
        })  # type: ignore[arg-type]

    @property
    def reference_plan_digest(self) -> str:
        return self.reference_execution.plan_digest

    @property
    def reference_session_digest(self) -> str:
        return self.reference_execution.session_digest

    @property
    def digest(self) -> str:
        return canonical_digest(self._domain, self.to_dict())


QualificationAttempt = CohortQualificationAttempt | DiscoveryQualificationAttempt

def _planned_prompt_digests(prepared: PreparedMarginalRuntime) -> tuple[str, ...]:
    plan = prepared.baseline_session_plan
    workload = marginal_workload_digest(plan)
    rows = []
    for batch_index, prompts in enumerate(plan.prompt_batches):
        for prompt_index, prompt in enumerate(prompts):
            rows.append(canonical_digest(
                "optima.qualification.prompt-occurrence",
                {
                    "batch_index": batch_index,
                    "prompt_index": prompt_index,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "workload_digest": workload,
                },
            ))
    return tuple(sorted(rows))

def _validate_pre_execution(
    value: CausalQualificationInput,
) -> tuple[CalibrationManifest, dict[str, GraphVerificationGrade]]:
    if type(value) is not CausalQualificationInput:
        raise QualificationRunnerError("qualification input has the wrong type")
    reference = value.candidates[0].profile.reference
    workload = marginal_workload_digest(value.prepared.baseline_session_plan)
    secret_commitment = hashlib.sha256(
        b"optima-selection-secret-v1\0" + value.selection_secret
    ).hexdigest()
    if (
        value.commitment.source_plan_digest != value.prepared.source.digest
        or value.commitment.reference_manifest_digest != reference.digest
        or value.commitment.workload_digest != workload
        or reference.workload_digest != workload
        or value.commitment.prompt_digests != _planned_prompt_digests(value.prepared)
        or value.commitment.entropy_source_digest != reference.selection_policy_digest
        or value.commitment.secret_commitment != secret_commitment
        or value.pristine_stack.digest != reference.pristine_stack_digest
        or value.pristine_launch.digest != reference.pristine_launch_digest
        or value.reference_engine_config.digest
        != value.reference_preflight.engine_config_digest
        or value.reference_preflight.launch_digest != value.pristine_launch.digest
    ):
        raise QualificationRunnerError("pre-execution commitment/reference identity differs")
    calibration = reopen_calibration_evidence(
        value.evidence_root,
        value.calibration_artifact_ref,
        expected_threshold_policy=value.calibration_threshold_policy,
        expected_manifest=value.calibration_manifest,
        expected_context=value.calibration_context,
    )
    grades: dict[str, GraphVerificationGrade] = {}
    for prepared, authority in zip(
        value.prepared.candidates, value.candidates, strict=True
    ):
        profile = authority.profile
        if (
            profile.reference != reference
            or profile.calibration_digest != calibration.digest
            or profile.calibration_context_digest != value.calibration_context.digest
            or profile.runtime_resource_policy_digest
            != value.expected_runtime_resource_policy_digest
            or (profile.tokens_per_prompt, profile.topk_width)
            != (
                value.prepared.baseline_session_plan.max_new_tokens,
                value.prepared.baseline_session_plan.top_logprobs_num,
            )
        ):
            raise QualificationRunnerError("candidate qualification profile differs from cohort")
        if type(authority) is CandidateQualificationAuthority:
            grades[authority.selected_delta_digest] = reopen_graph_verification(
                value.evidence_root,
                authority.graph_artifact_ref,
                authority.graph_requirement,
                authority.graph_evidence_ref,
            )
            continue
        from optima.discovery import DiscoveryArmPlan

        if type(authority) is not DiscoveryCandidateQualificationAuthority or type(
            prepared.arm
        ) is not DiscoveryArmPlan:
            raise QualificationRunnerError("discovery pre-execution authority is mismatched")
        requirement = authority.execution_requirement
        arm, launch = prepared.arm, prepared.launch
        expected = (
            arm.digest,
            arm.proposal_digest,
            arm.selected_delta_digest,
            arm.candidate_stack_digest,
            arm.candidate_tree_digest,
            launch.digest,
            prepared.binding.launch_binding.native_build_spec.digest,
            arm.policy_digest,
            arm.build_profile_digest,
            launch.worker_distribution_digest,
            launch.engine_config_digest,
            launch.hardware.tp_size,
        )
        actual = (
            requirement.arm_digest,
            requirement.proposal_digest,
            requirement.selected_delta_digest,
            requirement.candidate_stack_digest,
            requirement.candidate_tree_digest,
            requirement.candidate_launch_digest,
            requirement.native_build_spec_digest,
            requirement.discovery_policy_digest,
            requirement.build_profile_digest,
            requirement.worker_distribution_digest,
            requirement.engine_config_digest,
            requirement.expected_tp_size,
        )
        if (
            expected != actual
            or profile.execution_requirement_digest != requirement.digest
            or value.calibration_context.verification_policy_digest
            != requirement.activation_policy_digest
        ):
            raise QualificationRunnerError(
                "discovery requirement differs from the prepared runtime"
            )
    return calibration, grades

def _selected_frames(
    lifecycle: MarginalLifecycleEvidence,
    selected_delta_digest: str,
    prompts: tuple[str, ...],
) -> tuple[tuple[str, str, tuple[dict[str, object], ...]], ...]:
    candidates = tuple(row.arm.selected_delta_digest for row in lifecycle.candidates)
    if candidates.count(selected_delta_digest) != 1:
        raise QualificationRunnerError("selected candidate is absent or ambiguous")
    candidate_index = candidates.index(selected_delta_digest) + 1
    by_prompt = dict(_trajectory_rows(lifecycle)[1])
    texts = _selected_prompt_texts(lifecycle)
    result = []
    for prompt in prompts:
        if prompt not in by_prompt or prompt not in texts:
            raise QualificationRunnerError("selected prompt is absent from the lifecycle")
        frames = by_prompt[prompt]
        result.append((prompt, texts[prompt], tuple(frames[index] for index in (0, candidate_index, -1))))
    return tuple(result)

def _reference_request(
    lifecycle: MarginalLifecycleEvidence,
    authority: CandidateAuthority,
    selection: SelectionReceipt,
    *,
    session_id: str,
    plan_digest: str,
    request_id: str,
    nonce: str,
    index: int,
) -> ReferenceRequest:
    prompts = []
    for prompt_digest, prompt_text, frames in _selected_frames(
        lifecycle, authority.selected_delta_digest, selection.selected_prompt_digests
    ):
        roles = []
        for frame in frames:
            supports = tuple(
                tuple(sorted(row[1] for row in position))
                for position in frame["top_logprobs"]
            )
            roles.append(ReferenceRoleInput(tuple(frame["output_ids"]), supports))
        prompts.append(ReferencePromptInput(prompt_digest, prompt_text, tuple(roles)))
    return ReferenceRequest(
        session_id,
        authority.profile.reference.pristine_launch_digest,
        plan_digest,
        request_id,
        nonce,
        index,
        authority.profile.tokens_per_prompt,
        authority.profile.topk_width,
        tuple(prompts),
    )

def _task_digests(
    profile: QualificationProfile | DiscoveryQualificationProfile,
    prompt_digest: str,
) -> tuple[str, ...]:
    return tuple(sorted(canonical_digest(
        "optima.qualification.hidden-task",
        {
            "corpus": profile.reference.hidden_corpus_commitment,
            "judge": profile.reference.hidden_judge_digest,
            "policy": profile.hidden_task_policy_digest,
            "prompt": prompt_digest,
            "index": index,
        },
    ) for index in range(profile.hidden_tasks_per_prompt)))

def _hidden_judge_binding(
    profile: QualificationProfile | DiscoveryQualificationProfile,
) -> HiddenJudgeBinding:
    return HiddenJudgeBinding(
        profile.reference.hidden_corpus_commitment,
        profile.reference.hidden_judge_digest,
        profile.hidden_task_policy_digest,
    )

def _rollout(
    *,
    profile: QualificationProfile | DiscoveryQualificationProfile,
    prompt_digest: str,
    frame: dict[str, object],
    role_input: ReferenceRoleInput,
    role_evidence: object,
    hidden_judge: HiddenJudge,
) -> RawRolloutEvidence:
    tokens = []
    evidence_tokens = tuple(getattr(role_evidence, "tokens"))
    for position, (output_id, support, teacher, raw_position) in enumerate(zip(
        role_input.output_ids,
        role_input.supports,
        evidence_tokens,
        frame["top_logprobs"],
        strict=True,
    )):
        by_token = {row[1]: float(row[0]) for row in raw_position}
        if tuple(sorted(by_token)) != support:
            raise QualificationRunnerError("rollout support differs from its T request")
        rollout = distribution_from_f32_logprobs(
            support,
            tuple(by_token[token] for token in support),
            true_argmax_token_id=raw_position[0][1],
        )
        teacher_distribution = distribution_from_f32_logprobs(
            support,
            teacher.support_logprobs,
            true_argmax_token_id=teacher.true_argmax_token_id,
        )
        tokens.append(RawTokenEvidence(
            position,
            output_id,
            target_nll_from_f32(teacher.target_logprob),
            teacher_distribution,
            rollout,
        ))
    tasks = _task_digests(profile, prompt_digest)
    receipt = hidden_judge(
        prompt_digest=prompt_digest,
        output_ids=role_input.output_ids,
        task_digests=tasks,
    )
    if (
        type(receipt) is not HiddenJudgeReceipt
        or receipt.binding_digest != _hidden_judge_binding(profile).digest
        or receipt.prompt_digest != prompt_digest
        or receipt.output_ids_digest
        != hidden_judge_output_digest(prompt_digest, role_input.output_ids)
        or receipt.task_digests != tasks
    ):
        raise QualificationRunnerError("hidden judge receipt differs from the sealed rollout")
    return RawRolloutEvidence(
        tuple(tokens),
        tuple(
            RawHiddenTaskResult(task, passed)
            for task, passed in zip(tasks, receipt.passed, strict=True)
        ),
    )

def _raw_artifact(
    lifecycle: MarginalLifecycleEvidence,
    authority: CandidateAuthority,
    calibration: CalibrationManifest,
    selection: SelectionReceipt,
    reference_execution: PristineReferenceExecutionEvidence,
    exchange: object,
    hidden_judge: HiddenJudge,
) -> ReferenceQualityRawArtifact:
    profile = authority.profile
    request = exchange.request
    request_digest = exchange.request_sha256
    frames = _selected_frames(
        lifecycle, authority.selected_delta_digest, selection.selected_prompt_digests
    )
    prompts = []
    for (prompt_digest, _text, role_frames), request_prompt, teacher_prompt in zip(
        frames, request.prompts, exchange.evidence.prompts, strict=True
    ):
        if request_prompt.prompt_digest != prompt_digest or teacher_prompt.prompt_digest != prompt_digest:
            raise QualificationRunnerError("T prompt was relabeled")
        rows = tuple(_rollout(
            profile=profile,
            prompt_digest=prompt_digest,
            frame=frame,
            role_input=role_input,
            role_evidence=role_evidence,
            hidden_judge=hidden_judge,
        ) for frame, role_input, role_evidence in zip(
            role_frames, request_prompt.roles, teacher_prompt.roles, strict=True
        ))
        prompts.append(RawPromptQualityEvidence(prompt_digest, *rows))
    lifecycle_digest = candidate_lifecycle_digest(
        lifecycle, selected_delta_digest=authority.selected_delta_digest
    )
    identity = qualification_identity_digest(
        profile,
        graph_requirement=_requirement(authority),
        selection=selection,
        calibration=calibration,
        candidate_lifecycle=lifecycle_digest,
        t_session=reference_execution.session,
        t_request_sha256=request_digest,
        selected_delta_digest=authority.selected_delta_digest,
    )
    binding = ReferenceQualityRawBinding(
        identity,
        profile.reference.digest,
        calibration.digest,
        selection.digest,
        lifecycle_digest,
        selected_trajectory_digest(
            lifecycle,
            selected_delta_digest=authority.selected_delta_digest,
            selected_prompt_digests=selection.selected_prompt_digests,
        ),
        selected_trajectory_projection_digest(
            lifecycle,
            selected_delta_digest=authority.selected_delta_digest,
            selected_prompt_digests=selection.selected_prompt_digests,
        ),
        selection.selected_prompt_digests,
        reference_execution.session.digest,
        request_digest,
        profile.support_policy_digest,
        derived_hidden_task_plan_digest(profile, selection.selected_prompt_digests),
        profile.nll_tail_threshold,
        profile.tokens_per_prompt,
        profile.topk_width,
        profile.hidden_tasks_per_prompt,
    )
    return ReferenceQualityRawArtifact(binding, tuple(prompts))

def _speed_decision(speed: MarginalSpeedProjection) -> QualificationDecision:
    if not speed.verdict.confident:
        return QualificationDecision.NO_DECISION
    return QualificationDecision.PASS if speed.verdict.passed_speedup else QualificationDecision.FAIL

def _report_reason(
    graph: GraphVerificationGrade,
    speed: QualificationDecision,
    quality: ReferenceQualityVerdict,
) -> str:
    if graph.decision is not QualificationDecision.PASS:
        return graph.reason
    if speed is QualificationDecision.NO_DECISION:
        return "speed_noise"
    if speed is QualificationDecision.FAIL:
        return "speed_regression"
    if quality.decision == "FAIL":
        return "quality_regression"
    if quality.decision == "NO_DECISION":
        return "quality_overlap"
    return "qualified"


def _discovery_report_reason(
    execution: DiscoveryExecutionGrade,
    speed: QualificationDecision,
    quality: ReferenceQualityVerdict,
) -> str:
    if execution.decision is not QualificationDecision.PASS:
        return execution.reason
    if speed is QualificationDecision.NO_DECISION:
        return "speed_noise"
    if speed is QualificationDecision.FAIL:
        return "speed_regression"
    if quality.decision == "FAIL":
        return "quality_regression"
    if quality.decision == "NO_DECISION":
        return "quality_overlap"
    return "qualified"

def qualification_authority_digest(value: CausalQualificationInput) -> str:
    """Bind the durable attempt to all validator-owned inputs available before B."""

    if all(type(row) is CandidateQualificationAuthority for row in value.candidates):
        return canonical_digest("optima.qualification.causal-authority", {
        "calibration": [value.calibration_threshold_policy.digest, value.calibration_manifest.digest,
                        value.calibration_context.digest, value.calibration_artifact_ref.to_dict()],
        "candidates": [{
            "arm": prepared.arm.digest, "delta": authority.selected_delta_digest,
            "graph_artifact": authority.graph_artifact_ref.to_dict(),
            "graph_evidence_ref": authority.graph_evidence_ref.digest,
            "graph_requirement": authority.graph_requirement.digest,
            "launch": prepared.launch.digest, "profile": authority.profile.digest,
            "native": prepared.binding.launch_binding.native_build_spec.digest,
            "preflight": prepared.binding.launch_binding.runtime_preflight_receipt.sha256,
            "target": prepared.arm.transition.target_id,
        } for prepared, authority in zip(value.prepared.candidates, value.candidates, strict=True)],
        "commitment": value.commitment.digest, "model_mount": value.model_mount.digest,
        "incumbent_preflight": value.prepared.incumbent_binding.launch_binding.runtime_preflight_receipt.sha256,
        "policies": [value.expected_launch_resource_policy_digest,
                     value.expected_runtime_resource_policy_digest, value.expected_device_policy_digest],
        "reference": [value.pristine_stack.digest, value.pristine_launch.digest,
                      value.pristine_binding.native_build_spec.digest,
                      value.pristine_binding.controller_distribution_digest,
                      value.pristine_binding.runtime_preflight_receipt.sha256,
                      value.reference_engine_config.digest, value.reference_preflight.digest],
        "source": value.prepared.source.digest,
        })
    if len(value.candidates) != 1 or type(
        value.candidates[0]
    ) is not DiscoveryCandidateQualificationAuthority:
        raise QualificationRunnerError("qualification authority mode is inconsistent")
    authority = value.candidates[0]
    prepared = value.prepared.candidates[0]
    return canonical_digest("optima.qualification.discovery-causal-authority", {
        "calibration": [value.calibration_threshold_policy.digest,
                        value.calibration_manifest.digest,
                        value.calibration_context.digest,
                        value.calibration_artifact_ref.to_dict()],
        "candidate": {
            "arm": prepared.arm.digest,
            "delta": authority.selected_delta_digest,
            "execution_requirement": authority.execution_requirement.digest,
            "launch": prepared.launch.digest,
            "native": prepared.binding.launch_binding.native_build_spec.digest,
            "preflight": prepared.binding.launch_binding.runtime_preflight_receipt.sha256,
            "profile": authority.profile.digest,
            "proposal": authority.execution_requirement.proposal_digest,
        },
        "commitment": value.commitment.digest,
        "model_mount": value.model_mount.digest,
        "incumbent_preflight": (
            value.prepared.incumbent_binding.launch_binding
            .runtime_preflight_receipt.sha256
        ),
        "policies": [value.expected_launch_resource_policy_digest,
                     value.expected_runtime_resource_policy_digest,
                     value.expected_device_policy_digest],
        "reference": [value.pristine_stack.digest, value.pristine_launch.digest,
                      value.pristine_binding.native_build_spec.digest,
                      value.pristine_binding.controller_distribution_digest,
                      value.pristine_binding.runtime_preflight_receipt.sha256,
                      value.reference_engine_config.digest,
                      value.reference_preflight.digest],
        "source": value.prepared.source.digest,
    })

def _validate_reference_execution(
    attempt: QualificationAttempt, expected: CausalQualificationInput
) -> None:
    witness = attempt.reference_execution
    identity = runtime_identity_from_preflight(expected.pristine_binding.runtime_preflight_receipt)
    runtime_digest = canonical_digest("optima.qualification.reference-runtime", {
        "base": identity.base_engine_digest, "runtime": identity.runtime_digest,
        "validator_overlay": identity.validator_overlay_digest,
    })
    reference = expected.candidates[0].profile.reference
    plan_digest = canonical_digest("optima.eval.reference-session-plan", {
        "engine_config_digest": expected.reference_engine_config.digest,
        "expected_preflight_digest": expected.reference_preflight.digest,
        "pristine_stack_digest": expected.pristine_stack.digest,
        "reference_manifest_digest": reference.digest,
        "request_plan_digest": witness.request_plan_digest,
        "request_sha256": list(witness.request_sha256),
    })
    session_digest = canonical_digest("optima.eval.pristine-reference-session", {
        "exchanges": [{"request_index": index, "request_sha256": request,
                       "evidence_frame_sha256": evidence}
                      for index, (request, evidence) in enumerate(zip(
                          witness.request_sha256, witness.evidence_frame_sha256, strict=True))],
        "launch_digest": witness.launch_digest,
        "preflight_digest": expected.reference_preflight.digest,
        "reference_manifest_digest": reference.digest,
        "request_plan_digest": witness.request_plan_digest,
        "schema": "optima.pristine-reference-session.v1", "session_id": witness.session_id,
        "session_plan_digest": witness.plan_digest,
    })
    pre, post = witness.device_receipts
    if (
        (witness.launch_digest, witness.runtime_identity_digest,
         witness.runtime_preflight_receipt_sha256, witness.arena_model_receipt_digest,
         witness.resource_policy_digest, witness.build_spec_digest, witness.plan_digest,
         witness.session_digest, witness.request_sha256)
        != (expected.pristine_launch.digest, runtime_digest,
            expected.pristine_binding.runtime_preflight_receipt.sha256, expected.model_mount.digest,
            expected.expected_runtime_resource_policy_digest,
            expected.pristine_binding.native_build_spec.digest, plan_digest, session_digest,
            tuple(row.t_request_sha256 for row in attempt.reports))
        or pre[2] != post[2] or pre[1] >= post[1] or pre[4] != post[4]
        or pre[5:7] != post[5:7] or pre[6] != expected.expected_device_policy_digest
        or len(pre[4]) != expected.pristine_launch.hardware.visible_gpu_count
        or float(pre[7]) < attempt.entropy_observed_monotonic_s
        or float(pre[8]) > float(post[7])
        or float(post[8]) > attempt.teardown_after_t.observed_monotonic_s
    ):
        raise QualificationRunnerError("pristine T execution witness differs from causal authority")

def publish_causal_qualification(
    root: Path, attempt: QualificationAttempt
) -> EvidenceArtifactRef:
    if type(attempt) is CohortQualificationAttempt:
        domain, schema = ATTEMPT_DOMAIN, ATTEMPT_SCHEMA
    elif type(attempt) is DiscoveryQualificationAttempt:
        domain, schema = DISCOVERY_ATTEMPT_DOMAIN, DISCOVERY_ATTEMPT_SCHEMA
    else:
        raise QualificationRunnerError("qualification attempt is not typed")
    return publish_evidence(
        root,
        canonical_json_bytes(attempt.to_dict()),
        domain=domain,
        media_type="application/json",
        schema=schema,
    )

def reopen_causal_qualification(
    root: Path, reference: EvidenceArtifactRef, *, expected: CausalQualificationInput
) -> QualificationAttempt:
    """Authenticate and independently regrade one durable cohort attempt."""

    try:
        discovery_mode = (
            len(expected.candidates) == 1
            and type(expected.candidates[0])
            is DiscoveryCandidateQualificationAuthority
        )
        artifact_type = (
            (DISCOVERY_ATTEMPT_DOMAIN, "application/json", DISCOVERY_ATTEMPT_SCHEMA)
            if discovery_mode
            else (ATTEMPT_DOMAIN, "application/json", ATTEMPT_SCHEMA)
        )
        if type(reference) is not EvidenceArtifactRef or (
            reference.domain, reference.media_type, reference.schema
        ) != artifact_type:
            raise QualificationRunnerError("qualification artifact type is not authoritative")
        payload = _canonical_payload(reopen_evidence(root, reference))
        attempt = (
            DiscoveryQualificationAttempt.from_dict(payload)
            if discovery_mode
            else CohortQualificationAttempt.from_dict(payload)
        )
        if attempt.to_dict() != payload:
            raise QualificationRunnerError("qualification artifact is not semantically canonical")
        if attempt.authority_digest != qualification_authority_digest(expected):
            raise QualificationRunnerError("qualification authority digest differs")
        attempt.selection.reopen(attempt.commitment, attempt.entropy)
        if (
            attempt.source_digest != expected.prepared.source.digest
            or attempt.commitment != expected.commitment
            or attempt.selection.sealed_cohort_trajectory_digest != attempt.cohort_trajectory_digest
            or tuple(row.selected_delta_digest for row in attempt.reports)
            != tuple(row.selected_delta_digest for row in expected.candidates)
        ):
            raise QualificationRunnerError("qualification cohort identity differs")
        _validate_reference_execution(attempt, expected)
        calibration = reopen_calibration_evidence(
            root, expected.calibration_artifact_ref,
            expected_threshold_policy=expected.calibration_threshold_policy,
            expected_manifest=expected.calibration_manifest,
            expected_context=expected.calibration_context,
        )
        for report, authority, prepared in zip(
            attempt.reports, expected.candidates, expected.prepared.candidates, strict=True
        ):
            raw = report.raw_quality_binding
            quality = score_reference_quality(
                reopen_reference_quality_evidence(root, report.raw_quality_artifact,
                                                  expected_binding=raw),
                calibration=calibration, expected_context=expected.calibration_context,
            )
            speed = report.speed_witness
            rates = speed.rates
            if (
                (speed.selected_delta_digest, speed.candidate_launch_digest,
                 speed.runtime_resource_policy_digest, speed.evidence_digest)
                != (authority.selected_delta_digest, prepared.launch.digest,
                    expected.expected_runtime_resource_policy_digest, report.speed_evidence_digest)
                or tuple(row.launch_digest for row in rates)
                != (expected.prepared.baseline_launch.digest, prepared.launch.digest,
                    expected.prepared.baseline_launch.digest)
                or len({row.session_id for row in rates}) != 3
            ):
                raise QualificationRunnerError("speed witness differs from its marginal arm")
            speed_grade, speedup = speed.regrade(
                calibration, expected.calibration_context
            )
            identity_common = {
                "calibration_digest": calibration.digest,
                "candidate_lifecycle_digest": raw.candidate_lifecycle_digest,
                "profile_digest": authority.profile.digest,
                "selected_delta_digest": authority.selected_delta_digest,
                "selection_digest": attempt.selection.digest,
                "t_request_sha256": report.t_request_sha256,
                "t_session_digest": attempt.reference_session_digest,
            }
            quality_grade = _decision(quality.decision)
            if type(authority) is CandidateQualificationAuthority:
                if type(report) is not CandidateQualificationReport:
                    raise QualificationRunnerError(
                        "registered qualification report type differs"
                    )
                graph = reopen_graph_verification(
                    root,
                    authority.graph_artifact_ref,
                    authority.graph_requirement,
                    authority.graph_evidence_ref,
                )
                identity = canonical_digest(
                    "optima.qualification.candidate-identity",
                    {
                        **identity_common,
                        "graph_requirement_digest": authority.graph_requirement.digest,
                    },
                )
                decision = _aggregate_decision(
                    graph.decision, speed_grade, quality_grade
                )
                headline = (
                    report.marginal_arm_digest, report.candidate_launch_digest,
                    report.target_id, report.profile_digest,
                    report.calibration_digest, report.graph_grade_digest,
                    report.graph_decision, report.speed_evidence_digest,
                    report.speed_decision, report.speedup,
                    report.quality_evidence_digest, report.quality_decision,
                    report.candidate_mean_teacher_nll, report.decision,
                    report.reason, report.retryable,
                )
                expected_headline = (
                    prepared.arm.digest, prepared.launch.digest,
                    prepared.arm.transition.target_id, authority.profile.digest,
                    calibration.digest, graph.digest, graph.decision,
                    report.speed_witness.evidence_digest, speed_grade, speedup,
                    quality.evidence_digest, quality_grade,
                    quality.candidate_mean_teacher_nll, decision,
                    _report_reason(graph, speed_grade, quality),
                    decision is QualificationDecision.NO_DECISION,
                )
            elif type(authority) is DiscoveryCandidateQualificationAuthority:
                if type(report) is not DiscoveryCandidateQualificationReport:
                    raise QualificationRunnerError(
                        "discovery qualification report type differs"
                    )
                execution = report.execution_grade
                reopen_discovery_execution_binding(
                    authority.execution_requirement,
                    execution,
                    prepared,
                    candidate_lifecycle_digest=raw.candidate_lifecycle_digest,
                    session_id=rates[1].session_id,
                )
                identity = canonical_digest(
                    "optima.qualification.discovery-candidate-identity",
                    {
                        **identity_common,
                        "execution_requirement_digest": (
                            authority.execution_requirement.digest
                        ),
                    },
                )
                decision = _aggregate_decision(
                    execution.decision, speed_grade, quality_grade
                )
                headline = (
                    report.discovery_arm_digest, report.proposal_digest,
                    report.candidate_launch_digest, report.profile_digest,
                    report.calibration_digest, report.execution_grade,
                    report.speed_evidence_digest, report.speed_decision,
                    report.speedup, report.quality_evidence_digest,
                    report.quality_decision, report.candidate_mean_teacher_nll,
                    report.decision, report.reason, report.retryable,
                )
                expected_headline = (
                    prepared.arm.digest,
                    authority.execution_requirement.proposal_digest,
                    prepared.launch.digest, authority.profile.digest,
                    calibration.digest, execution,
                    report.speed_witness.evidence_digest, speed_grade, speedup,
                    quality.evidence_digest, quality_grade,
                    quality.candidate_mean_teacher_nll, decision,
                    _discovery_report_reason(execution, speed_grade, quality),
                    decision is QualificationDecision.NO_DECISION,
                )
            else:
                raise QualificationRunnerError(
                    "qualification authority has an unsupported type"
                )
            binding = (
                raw.qualification_identity_digest, raw.reference_manifest_digest,
                raw.calibration_digest, raw.selection_digest, raw.selected_prompt_digests,
                raw.t_session_digest, raw.t_request_sha256, raw.support_policy_digest,
                raw.hidden_task_plan_digest, raw.nll_tail_threshold, raw.tokens_per_prompt,
                raw.topk_width, raw.hidden_tasks_per_prompt,
            )
            expected_binding = (
                identity, authority.profile.reference.digest, calibration.digest,
                attempt.selection.digest, attempt.selection.selected_prompt_digests,
                attempt.reference_session_digest, report.t_request_sha256,
                authority.profile.support_policy_digest,
                derived_hidden_task_plan_digest(authority.profile, attempt.selection.selected_prompt_digests),
                authority.profile.nll_tail_threshold, authority.profile.tokens_per_prompt,
                authority.profile.topk_width, authority.profile.hidden_tasks_per_prompt,
            )
            if binding != expected_binding or headline != expected_headline:
                raise QualificationRunnerError("candidate qualification does not independently regrade")
        return attempt
    except QualificationRunnerError:
        raise
    except Exception as exc:
        raise QualificationRunnerError(f"qualification evidence cannot reopen: {exc}") from None

def run_causal_qualification(
    value: CausalQualificationInput,
    *,
    executor: OCIEngineExecutor,
    entropy_provider: EntropyProvider,
    hidden_judge: HiddenJudge,
    deadline: float,
    id_factory: Callable[[], str] | None = None,
) -> EvidenceArtifactRef:
    """Run one complete causal cohort or raise without a partial PASS."""

    if type(executor) is not OCIEngineExecutor or not callable(entropy_provider) or not callable(hidden_judge):
        raise QualificationRunnerError("runner authorities are not exact and callable")
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline) or deadline <= 0:
        raise QualificationRunnerError("deadline must be finite and positive")
    make_id = id_factory or (lambda: secrets.token_hex(16))
    try:
        judge_binding = hidden_judge.binding
    except (AttributeError, TypeError, ValueError) as exc:
        raise QualificationRunnerError("hidden judge has no typed authority binding") from exc
    expected_judge_bindings = tuple(
        _hidden_judge_binding(authority.profile) for authority in value.candidates
    )
    if (
        type(judge_binding) is not HiddenJudgeBinding
        or any(binding != expected_judge_bindings[0] for binding in expected_judge_bindings)
        or judge_binding != expected_judge_bindings[0]
    ):
        raise QualificationRunnerError("hidden judge authority differs from the sealed cohort")
    calibration, graph_grades = _validate_pre_execution(value)
    judge_cache: dict[tuple[object, ...], HiddenJudgeReceipt] = {}

    def judge_once(*, prompt_digest: str, output_ids: tuple[int, ...],
                   task_digests: tuple[str, ...]) -> HiddenJudgeReceipt:
        key = (judge_binding.digest, prompt_digest,
               hidden_judge_output_digest(prompt_digest, output_ids), task_digests)
        if key not in judge_cache:
            judge_cache[key] = hidden_judge(prompt_digest=prompt_digest, output_ids=output_ids,
                                            task_digests=task_digests)
        return judge_cache[key]

    with executor.exclusive_transaction():
        lifecycle = run_marginal_lifecycle(
            value.prepared,
            executor=executor,
            model_mount=value.model_mount,
            deadline=float(deadline),
        )
        discovery_grades: dict[str, DiscoveryExecutionGrade] = {}
        for authority in value.candidates:
            if type(authority) is DiscoveryCandidateQualificationAuthority:
                discovery_grades[authority.selected_delta_digest] = (
                    grade_discovery_execution(
                        authority.execution_requirement, lifecycle
                    )
                )
        teardown_before = executor.prove_quiescent()
        last_post = lifecycle.baseline_after.device_receipts[-1].completed_monotonic_s
        if teardown_before.observed_monotonic_s < last_post:
            raise QualificationRunnerError("pre-T quiescence predates B-prime teardown")
        entropy = entropy_provider(value.commitment, teardown_before)
        if type(entropy) is not SelectionEntropyReceipt:
            raise QualificationRunnerError("entropy provider returned an untyped receipt")
        entropy_observed = float(executor.manager.clock())
        if not math.isfinite(entropy_observed) or entropy_observed < teardown_before.observed_monotonic_s:
            raise QualificationRunnerError("entropy observation predates teardown")
        selection = SelectionReceipt.reveal(
            value.commitment,
            secret=value.selection_secret,
            entropy=entropy,
            sealed_cohort_trajectory_digest=cohort_trajectory_digest(lifecycle),
        )
        request_plan_digest = canonical_digest(
            "optima.qualification.reference-request-plan",
            {
                "candidate_deltas": [row.selected_delta_digest for row in value.candidates],
                "cohort_trajectory_digest": cohort_trajectory_digest(lifecycle),
                "reference_manifest_digest": value.candidates[0].profile.reference.digest,
                "selection_digest": selection.digest,
            },
        )
        session_id = make_id()
        requests = tuple(_reference_request(
            lifecycle,
            authority,
            selection,
            session_id=session_id,
            plan_digest=request_plan_digest,
            request_id=make_id(),
            nonce=make_id(),
            index=index,
        ) for index, authority in enumerate(value.candidates))
        plan = ReferenceSessionPlan(
            value.candidates[0].profile.reference,
            value.pristine_stack,
            value.reference_engine_config.digest,
            value.reference_engine_config,
            value.reference_preflight,
            request_plan_digest,
            requests,
        )
        reference_execution = executor.execute_reference(
            value.pristine_launch,
            value.pristine_binding,
            value.model_mount,
            plan,
            deadline=float(deadline),
        )
        teardown_after = executor.prove_quiescent()
        t_pre, t_post = reference_execution.device_receipts
        if (
            t_pre.started_monotonic_s < entropy_observed
            or t_post.completed_monotonic_s > teardown_after.observed_monotonic_s
        ):
            raise QualificationRunnerError("pristine T does not lie between causal boundaries")

    reports = []
    exchanges = reference_execution.session.exchanges
    if (
        len(exchanges) != len(value.candidates)
        or tuple(row.request for row in exchanges) != requests
        or tuple(row.request_sha256 for row in exchanges)
        != tuple(request_sha256(row) for row in requests)
    ):
        raise QualificationRunnerError("T exchange coverage differs from the candidate cohort")
    for authority, exchange in zip(value.candidates, exchanges, strict=True):
        raw = _raw_artifact(
            lifecycle,
            authority,
            calibration,
            selection,
            reference_execution,
            exchange,
            judge_once,  # type: ignore[arg-type]
        )
        validate_quality_binding(
            authority.profile,
            raw,
            lifecycle,
            selected_delta_digest=authority.selected_delta_digest,
            commitment=value.commitment,
            entropy=entropy,
            selection=selection,
            calibration=calibration,
            graph_requirement=_requirement(authority),
            reference_execution=reference_execution,
            reference_request_sha256=exchange.request_sha256,
        )
        raw_ref = publish_evidence(
            value.evidence_root,
            canonical_json_bytes(raw.to_dict()),
            domain=RAW_QUALITY_DOMAIN,
            media_type="application/json",
            schema=RAW_QUALITY_SCHEMA,
        )
        quality_evidence = reopen_reference_quality_evidence(
            value.evidence_root,
            raw_ref,
            expected_binding=raw.binding,
        )
        quality = score_reference_quality(
            quality_evidence,
            calibration=calibration,
            expected_context=value.calibration_context,
        )
        speed = project_marginal_speed(
            lifecycle,
            selected_delta_digest=authority.selected_delta_digest,
            calibration=calibration,
            expected_context=value.calibration_context,
            model_mount=value.model_mount,
            expected_launch_resource_policy_digest=value.expected_launch_resource_policy_digest,
            expected_runtime_resource_policy_digest=value.expected_runtime_resource_policy_digest,
            expected_device_policy_digest=value.expected_device_policy_digest,
        )
        speed_witness = SpeedWitness.from_projection(
            speed, value.expected_runtime_resource_policy_digest
        )
        speed_grade = _speed_decision(speed)
        quality_grade = _decision(quality.decision)
        candidate = next(
            row for row in lifecycle.candidates
            if row.arm.selected_delta_digest == authority.selected_delta_digest
        )
        if type(authority) is CandidateQualificationAuthority:
            graph = graph_grades[authority.selected_delta_digest]
            decision = _aggregate_decision(
                graph.decision, speed_grade, quality_grade
            )
            reports.append(CandidateQualificationReport(
                authority.selected_delta_digest,
                candidate.arm.digest,
                candidate.candidate.launch.digest,
                candidate.arm.transition.target_id,
                authority.profile.digest,
                calibration.digest,
                graph.digest,
                graph.decision,
                speed.evidence_digest,
                speed_grade,
                format(speed.verdict.speedup, ".17g"),
                quality.evidence_digest,
                quality_grade,
                quality.candidate_mean_teacher_nll,
                raw_ref,
                raw.binding,
                speed_witness,
                exchange.request_sha256,
                decision,
                _report_reason(graph, speed_grade, quality),
                decision is QualificationDecision.NO_DECISION,
            ))
        else:
            if type(authority) is not DiscoveryCandidateQualificationAuthority:
                raise QualificationRunnerError(
                    "candidate authority has an unsupported type"
                )
            execution = discovery_grades[authority.selected_delta_digest]
            decision = _aggregate_decision(
                execution.decision, speed_grade, quality_grade
            )
            reports.append(DiscoveryCandidateQualificationReport(
                authority.selected_delta_digest,
                candidate.arm.digest,
                authority.execution_requirement.proposal_digest,
                candidate.candidate.launch.digest,
                authority.profile.digest,
                calibration.digest,
                execution,
                speed.evidence_digest,
                speed_grade,
                format(speed.verdict.speedup, ".17g"),
                quality.evidence_digest,
                quality_grade,
                quality.candidate_mean_teacher_nll,
                raw_ref,
                raw.binding,
                speed_witness,
                exchange.request_sha256,
                decision,
                _discovery_report_reason(execution, speed_grade, quality),
                decision is QualificationDecision.NO_DECISION,
            ))
    attempt_type = (
        DiscoveryQualificationAttempt
        if all(
            type(row) is DiscoveryCandidateQualificationReport for row in reports
        )
        else CohortQualificationAttempt
    )
    attempt = attempt_type(
        qualification_authority_digest(value),
        value.prepared.source.digest,
        cohort_trajectory_digest(lifecycle),
        value.commitment,
        teardown_before,
        entropy,
        entropy_observed,
        selection,
        ReferenceExecutionWitness.from_execution(reference_execution, plan.digest),
        teardown_after,
        tuple(reports),
    )
    reference = publish_causal_qualification(value.evidence_root, attempt)
    reopen_causal_qualification(value.evidence_root, reference, expected=value)
    return reference

__all__ = [
    "CandidateQualificationAuthority", "CandidateQualificationReport",
    "DiscoveryCandidateQualificationAuthority",
    "DiscoveryCandidateQualificationReport", "DiscoveryQualificationAttempt",
    "CausalQualificationInput", "CohortQualificationAttempt", "EntropyProvider",
    "HiddenJudge", "HiddenJudgeBinding", "HiddenJudgeReceipt", "QualificationRunnerError",
    "ReferenceExecutionWitness", "SpeedWitness", "hidden_judge_output_digest",
    "publish_causal_qualification", "qualification_authority_digest",
    "reopen_causal_qualification", "run_causal_qualification",
]
