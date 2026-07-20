"""Causal B/C/B-prime to pristine-T qualification authority.

This module joins already-authenticated component evidence.  It does not load a
candidate, choose an incumbent, crown a winner, or mutate settlement state.
"""

from __future__ import annotations

import hashlib
import json
import math
import secrets
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Callable, ClassVar, Protocol

from optima.eval.calibration import (
    CalibrationContext, CalibrationManifest, CalibrationThresholdPolicy,
    decimal_value, reopen_calibration_evidence,
)
from optima.eval.crossover_runtime import (
    CrossoverRuntimeError,
    ResidentCrossoverEvidence,
    ResidentCrossoverPlan,
    ResidentMarginalLifecycleEvidence,
    ResidentReadRate,
    ResidentSpeedPolicy,
    SpeedStageDecision,
    run_resident_crossover_speed,
)
from optima.eval.engine_launch import EngineLaunchSpec, TrustedLaunchBinding
from optima.eval.evidence_store import EvidenceArtifactRef, publish_evidence, reopen_evidence
from optima.eval.device_state import DeviceStateReceipt
from optima.eval.marginal_runtime import (
    MarginalLifecycleEvidence, PreparedMarginalRuntime, run_marginal_lifecycle,
)
from optima.eval.oci_backend import (
    EngineExecutionEvidence, OCIEngineExecutor, PristineReferenceExecutionEvidence,
    TrustedArenaModelMountReceipt, runtime_identity_from_preflight,
)
from optima.eval.oci_outer_session import SessionExecutionPlan
from optima.eval.oci_process import OCIQuiescenceReceipt
from optima.eval.oci_reference_session import ReferenceSessionPlan
from optima.eval.oci_session_protocol import (
    AuditReceiptFacts, EngineSessionConfig, RuntimePreflightFacts, SlotAuditPolicy,
)
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
from optima._strict import require_exact_fields

class QualificationRunnerError(RuntimeError):
    """Infrastructure/authority failure; callers must treat it as NO_DECISION."""

    decision = QualificationDecision.NO_DECISION
    retryable = True

ATTEMPT_DOMAIN = "qualification.cohort-attempt"
ATTEMPT_SCHEMA = "optima.qualification.cohort-attempt.v1"
ATTEMPT_SCHEMA_V2 = "optima.qualification.cohort-attempt.v2"
ATTEMPT_SCHEMA_V3 = "optima.qualification.cohort-attempt.v3"
DISCOVERY_ATTEMPT_DOMAIN = "qualification.discovery-attempt"
DISCOVERY_ATTEMPT_SCHEMA = "optima.qualification.discovery-attempt.v1"
DISCOVERY_ATTEMPT_SCHEMA_V2 = "optima.qualification.discovery-attempt.v2"
STAGE_EXIT_DOMAIN = "qualification.stage-exit"
STAGE_EXIT_SCHEMA = "optima.qualification.stage-exit.v1"

LEGACY_SPEED_ESTIMATOR = "bcbp-baseline-range.v1"
REPEAT_SPEED_ESTIMATOR = "bcbpcbpp-max-arm-range.v1"
RESIDENT_SPEED_ESTIMATOR = "resident-adaptive-bcbp-v1"


@dataclass(frozen=True)
class SpeedEvidencePolicy:
    """Consensus identity for the speed-read shape and its estimator.

    Version 1 remains the calibrated production default and reopens historical
    B/C/B-prime artifacts byte-for-byte.  Version 2 is an explicit opt-in until a
    current-head GPU null/honest campaign calibrates it.  The second candidate
    read is not an optional runner knob; it is pre-B authority and therefore must
    agree across primary and reproduction.
    """

    version: int
    candidate_reads: int
    estimator: str

    def __post_init__(self) -> None:
        expected = {
            1: (1, LEGACY_SPEED_ESTIMATOR),
            2: (2, REPEAT_SPEED_ESTIMATOR),
            3: (0, RESIDENT_SPEED_ESTIMATOR),
        }
        if (
            type(self.version) is not int
            or self.version not in expected
            or (self.candidate_reads, self.estimator) != expected[self.version]
        ):
            raise QualificationRunnerError("speed evidence policy is unsupported")

    @classmethod
    def legacy(cls) -> "SpeedEvidencePolicy":
        return cls(1, 1, LEGACY_SPEED_ESTIMATOR)

    @classmethod
    def repeat(cls) -> "SpeedEvidencePolicy":
        return cls(2, 2, REPEAT_SPEED_ESTIMATOR)

    @classmethod
    def resident(cls) -> "SpeedEvidencePolicy":
        return cls(3, 0, RESIDENT_SPEED_ESTIMATOR)

    @property
    def digest(self) -> str:
        return canonical_digest("optima.qualification.speed-evidence-policy", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_reads": self.candidate_reads,
            "estimator": self.estimator,
            "version": self.version,
        }


# Keep production on the calibrated historical referee until a current-head GPU
# stock-null + honest-control campaign calibrates v2.  The complete repeat path is
# opt-in authority, never an unbound runner toggle.
DEFAULT_SPEED_EVIDENCE_POLICY = SpeedEvidencePolicy.legacy


class SpeedStageDisposition(str, Enum):
    """Pre-B authority for handling a non-passing resident speed control.

    Economic qualification is always terminal at a speed FAIL/NO_DECISION.  The
    calibration-observation disposition exists only for the validator's
    registered singleton bootstrap: its C arm is deliberately excluded from
    threshold derivation, so the already-authenticated B/C/B-prime lifecycle may
    continue to collect the stock quality observations.  The final report still
    retains the real non-passing speed grade and therefore cannot crown.
    """

    TERMINAL = "terminal"
    CALIBRATION_OBSERVATION = "calibration_observation"


def _strict(value: object, fields: set[str], label: str) -> dict[str, object]:
    return require_exact_fields(
        value, fields=frozenset(fields), label=label, error=QualificationRunnerError,
        exact_dict=True,
    )

def _encode_record(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    # Reports have a deliberately versioned wire shape: v1 omits the v2-only
    # repeat witness.  Let their custom serializer run even when nested inside
    # an attempt dataclass.
    if type(value) in {
        globals().get("CandidateQualificationReport"),
        globals().get("DiscoveryCandidateQualificationReport"),
        globals().get("ResidentSpeedWitness"),
    }:
        return value.to_dict()  # type: ignore[union-attr]
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
    audit_policies: tuple[SlotAuditPolicy, ...]
    # Legacy remains the production default until repeat-read is GPU-calibrated.
    # A v2 plan must opt in before B and is then authority-bound end to end.
    speed_evidence_policy: SpeedEvidencePolicy = field(
        default_factory=DEFAULT_SPEED_EVIDENCE_POLICY
    )
    resident_speed_plan: ResidentCrossoverPlan | None = None
    resident_audit_plan: SessionExecutionPlan | None = None
    speed_stage_disposition: SpeedStageDisposition = SpeedStageDisposition.TERMINAL

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
            or type(self.speed_evidence_policy) is not SpeedEvidencePolicy
            or type(self.speed_stage_disposition) is not SpeedStageDisposition
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
        audit_policies = tuple(self.audit_policies)
        if (
            len(audit_policies) != len(candidates)
            or any(type(row) is not SlotAuditPolicy for row in audit_policies)
            or any(
                row.expected_member_count != prepared.session_plan.engine_config.tp_size
                for row, prepared in zip(
                    audit_policies, self.prepared.candidates, strict=True
                )
            )
            or self.prepared.baseline_session_plan.audit_policy is not None
            or any(
                prepared.session_plan.audit_policy is not None
                for prepared in self.prepared.candidates
            )
        ):
            raise QualificationRunnerError(
                "slot audit authority differs from the candidate cohort"
            )
        object.__setattr__(self, "audit_policies", audit_policies)
        resident = self.resident_speed_plan
        if (self.speed_evidence_policy.version == 3) != (
            type(resident) is ResidentCrossoverPlan
        ):
            raise QualificationRunnerError(
                "resident speed plan coverage differs from speed policy"
            )
        resident_audit = self.resident_audit_plan
        if (self.speed_evidence_policy.version == 3) != (
            type(resident_audit) is SessionExecutionPlan
        ):
            raise QualificationRunnerError(
                "resident audit plan coverage differs from speed policy"
            )
        if resident is not None:
            if discovery or len(self.prepared.candidates) != 1:
                raise QualificationRunnerError(
                    "resident speed plan differs from qualification authority"
                )
            prepared_candidate = self.prepared.candidates[0]
            candidate_authority = self.candidates[0]
            assert resident_audit is not None
            if type(candidate_authority) is not CandidateQualificationAuthority:
                raise QualificationRunnerError(
                    "resident speed plan differs from qualification authority"
                )
            reference = candidate_authority.profile.reference
            baseline_common = set(
                self.prepared.baseline_launch.__dataclass_fields__
            ) - {"hardware", "resource_policy_digest"}
            baseline_binding_common = set(
                self.prepared.incumbent_binding.launch_binding.__dataclass_fields__
            ) - {"physical_hardware"}
            baseline_plan_common = set(
                self.prepared.baseline_session_plan.__dataclass_fields__
            ) - {"launch_digest", "expected_preflight"}
            baseline_preflight_common = set(
                self.prepared.baseline_session_plan.expected_preflight.__dataclass_fields__
            ) - {"launch_digest"}
            try:
                expected_resident_policy = ResidentSpeedPolicy.from_calibration(
                    max_stage_seconds=resident.policy.max_stage_seconds,
                    max_qualification_seconds=(
                        resident.policy.max_qualification_seconds
                    ),
                    calibration=self.calibration_manifest,
                    context=self.calibration_context,
                )
            except CrossoverRuntimeError as exc:
                raise QualificationRunnerError(str(exc)) from None
            resident_context = CalibrationContext(
                reference.digest,
                reference.arena_digest,
                reference.runtime_digest,
                reference.base_engine_digest,
                reference.model_revision_digest,
                reference.model_manifest_digest,
                reference.model_content_digest,
                reference.logical_hardware_digest,
                reference.workload_digest,
                candidate_authority.graph_requirement.binding.verification_policy_digest,
                reference.controller_distribution_digest,
            )
            if (
                resident.selected_delta_digest
                != prepared_candidate.arm.selected_delta_digest
                or reference.workload_digest
                != marginal_workload_digest(resident.baseline.session_plan)
                or any(
                    getattr(resident.baseline.launch, name)
                    != getattr(self.prepared.baseline_launch, name)
                    for name in baseline_common
                )
                or any(
                    getattr(resident.baseline.binding, name)
                    != getattr(
                        self.prepared.incumbent_binding.launch_binding, name
                    )
                    for name in baseline_binding_common
                )
                or any(
                    getattr(resident.baseline.session_plan, name)
                    != getattr(self.prepared.baseline_session_plan, name)
                    for name in baseline_plan_common
                )
                or any(
                    getattr(resident.baseline.session_plan.expected_preflight, name)
                    != getattr(
                        self.prepared.baseline_session_plan.expected_preflight,
                        name,
                    )
                    for name in baseline_preflight_common
                )
                or resident.candidate.launch.digest
                != prepared_candidate.launch.digest
                or resident.candidate.binding
                != prepared_candidate.binding.launch_binding
                or resident.candidate.session_plan
                != prepared_candidate.session_plan
                or resident.candidate.launch.resource_policy_digest
                != self.expected_launch_resource_policy_digest
                or self.pristine_launch.resource_policy_digest
                != self.expected_launch_resource_policy_digest
                or resident.policy != expected_resident_policy
                or self.calibration_context != resident_context
                or resident.candidate.runtime_resource_policy_digest
                != self.expected_runtime_resource_policy_digest
            ):
                raise QualificationRunnerError(
                    "resident speed plan differs from qualification authority"
                )
            if (
                resident_audit.launch_digest != prepared_candidate.launch.digest
                or resident_audit.expected_engine_config_digest
                != prepared_candidate.session_plan.expected_engine_config_digest
                or resident_audit.engine_config
                != prepared_candidate.session_plan.engine_config
                or resident_audit.expected_preflight
                != prepared_candidate.session_plan.expected_preflight
                or resident_audit.expected_discovery_overlay_identity_digest
                != prepared_candidate.session_plan.expected_discovery_overlay_identity_digest
                or resident_audit.audit_policy != audit_policies[0]
                or marginal_workload_digest(resident_audit)
                == marginal_workload_digest(prepared_candidate.session_plan)
            ):
                raise QualificationRunnerError(
                    "resident audit plan differs from qualification authority"
                )
        if (
            self.speed_stage_disposition
            is SpeedStageDisposition.CALIBRATION_OBSERVATION
            and (discovery or len(candidates) != 1 or resident is None)
        ):
            raise QualificationRunnerError(
                "calibration speed continuation requires a registered resident singleton"
            )
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

def _aggregate_decision(*values: QualificationDecision) -> QualificationDecision:
    if not values or any(type(row) is not QualificationDecision for row in values):
        raise QualificationRunnerError("aggregate decisions are not exactly typed")
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
    # 3 rates = the historical B/C/B-prime shape (digest-identical to all prior
    # witnesses); 5 rates = repeat-read evidence in RUN ORDER B, C, B', C', B''.
    rates: tuple[ChargedExecutionRate, ...]

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            if field != "rates":
                object.__setattr__(self, field, require_sha256_hex(getattr(self, field), field=field))
        rates = tuple(self.rates)
        if len(rates) not in (3, 5):
            raise QualificationRunnerError(
                "speed witness must contain B/C/B-prime rates (or B/C/B'/C'/B'' repeat reads)"
            )
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
        if (row.candidate_repeat is None) != (row.baseline_third is None):
            raise QualificationRunnerError("speed projection repeat-read evidence is unpaired")
        rates: tuple[ChargedExecutionRate, ...] = (
            row.baseline_before, row.candidate, row.baseline_after,
        )
        if row.candidate_repeat is not None and row.baseline_third is not None:
            rates = rates + (row.candidate_repeat, row.baseline_third)
        return cls(
            row.selected_delta_digest, row.candidate_launch_digest, row.calibration_digest,
            row.calibration_context_digest, row.workload_digest, runtime_policy,
            row.evidence_digest, rates,
        )

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    @property
    def policy(self) -> SpeedEvidencePolicy:
        # The read shape is inside the signed raw witness.  Keeping the policy
        # derived (rather than adding serialized fields) preserves legacy v1
        # witness bytes exactly.
        return (
            SpeedEvidencePolicy.legacy()
            if len(self.rates) == 3
            else SpeedEvidencePolicy.repeat()
        )

    @classmethod
    def from_dict(cls, value: object) -> "SpeedWitness":
        raw = _strict(value, set(cls.__dataclass_fields__), "speed witness")
        return cls(**{**raw, "rates": tuple(_rate_from_dict(row) for row in raw["rates"])})  # type: ignore[arg-type]

    def regrade(
        self,
        calibration: CalibrationManifest,
        context: CalibrationContext,
        *,
        expected_policy: SpeedEvidencePolicy | None = None,
    ) -> tuple[QualificationDecision, str]:
        if expected_policy is not None and (
            type(expected_policy) is not SpeedEvidencePolicy
            or self.policy != expected_policy
        ):
            raise QualificationRunnerError("speed witness policy differs from authority")
        if (
            self.calibration_digest != calibration.digest
            or self.calibration_context_digest != context.digest
            or self.workload_digest != context.workload_digest
            or not calibration.thresholds_frozen
        ):
            raise QualificationRunnerError("speed witness calibration authority differs")
        if len(self.rates) == 3:
            before, candidate, after = self.rates
            baseline_reads = [before.tokens_per_second, after.tokens_per_second]
            candidate_reads = [candidate.tokens_per_second]
        else:  # run order B, C, B', C', B''
            before, candidate, after, candidate_repeat, baseline_third = self.rates
            baseline_reads = [
                before.tokens_per_second, after.tokens_per_second,
                baseline_third.tokens_per_second,
            ]
            candidate_reads = [candidate.tokens_per_second, candidate_repeat.tokens_per_second]
        verdict = score_speedup(
            baseline_reads, candidate_reads,
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


def _resident_speed_projection_digest(
    *,
    selected_delta_digest: str,
    candidate_launch_digest: str,
    calibration_digest: str,
    calibration_context_digest: str,
    workload_digest: str,
    baseline_runtime_resource_policy_digest: str,
    candidate_runtime_resource_policy_digest: str,
    plan_digest: str,
    baseline_lane_digest: str,
    candidate_lane_digest: str,
    baseline_quiescence_digest: str,
    candidate_quiescence_digest: str,
    raw_crossover_digest: str,
    resident_policy: ResidentSpeedPolicy,
    rates: tuple[ResidentReadRate, ...],
    started_monotonic_s: float,
    completed_monotonic_s: float,
) -> str:
    return canonical_digest(
        "optima.qualification.resident-speed-witness.v1",
        {
            "baseline_lane": baseline_lane_digest,
            "baseline_quiescence": baseline_quiescence_digest,
            "calibration": calibration_digest,
            "calibration_context": calibration_context_digest,
            "candidate_lane": candidate_lane_digest,
            "candidate_launch": candidate_launch_digest,
            "candidate_quiescence": candidate_quiescence_digest,
            "plan": plan_digest,
            "policy": resident_policy.digest,
            "rates": [row.to_dict() for row in rates],
            "raw_crossover": raw_crossover_digest,
            "baseline_runtime_resource_policy": (
                baseline_runtime_resource_policy_digest
            ),
            "candidate_runtime_resource_policy": (
                candidate_runtime_resource_policy_digest
            ),
            "selected_delta": selected_delta_digest,
            "started_monotonic_s": format(started_monotonic_s, ".17g"),
            "completed_monotonic_s": format(completed_monotonic_s, ".17g"),
            "workload": workload_digest,
        },
    )


@dataclass(frozen=True)
class ResidentSpeedWitness:
    selected_delta_digest: str
    candidate_launch_digest: str
    calibration_digest: str
    calibration_context_digest: str
    workload_digest: str
    baseline_runtime_resource_policy_digest: str
    candidate_runtime_resource_policy_digest: str
    plan_digest: str
    baseline_lane_digest: str
    candidate_lane_digest: str
    baseline_quiescence_digest: str
    candidate_quiescence_digest: str
    raw_crossover_digest: str
    resident_policy: ResidentSpeedPolicy
    rates: tuple[ResidentReadRate, ...]
    started_monotonic_s: float
    completed_monotonic_s: float
    evidence_digest: str

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            if field not in {
                "resident_policy",
                "rates",
                "started_monotonic_s",
                "completed_monotonic_s",
            }:
                object.__setattr__(
                    self,
                    field,
                    require_sha256_hex(getattr(self, field), field=field),
                )
        if (
            type(self.resident_policy) is not ResidentSpeedPolicy
            or self.calibration_digest != self.resident_policy.calibration_digest
            or self.calibration_context_digest
            != self.resident_policy.calibration_context_digest
        ):
            raise QualificationRunnerError(
                "resident speed policy authority is inconsistent"
            )
        if (
            type(self.started_monotonic_s) is not float
            or type(self.completed_monotonic_s) is not float
            or not math.isfinite(self.started_monotonic_s)
            or not math.isfinite(self.completed_monotonic_s)
            or self.completed_monotonic_s <= self.started_monotonic_s
            or self.completed_monotonic_s - self.started_monotonic_s
            > self.resident_policy.max_stage_seconds
        ):
            raise QualificationRunnerError("resident speed wall time is invalid")
        rates = tuple(self.rates)
        expected_roles = {
            3: ("B", "C", "B_prime"),
            5: ("B", "C", "B_prime", "C_prime", "B_double_prime"),
        }
        if tuple(row.role for row in rates) != expected_roles.get(len(rates)):
            raise QualificationRunnerError("resident speed witness read order differs")
        object.__setattr__(self, "rates", rates)
        expected = _resident_speed_projection_digest(
            selected_delta_digest=self.selected_delta_digest,
            candidate_launch_digest=self.candidate_launch_digest,
            calibration_digest=self.calibration_digest,
            calibration_context_digest=self.calibration_context_digest,
            workload_digest=self.workload_digest,
            baseline_runtime_resource_policy_digest=(
                self.baseline_runtime_resource_policy_digest
            ),
            candidate_runtime_resource_policy_digest=(
                self.candidate_runtime_resource_policy_digest
            ),
            plan_digest=self.plan_digest,
            baseline_lane_digest=self.baseline_lane_digest,
            candidate_lane_digest=self.candidate_lane_digest,
            baseline_quiescence_digest=self.baseline_quiescence_digest,
            candidate_quiescence_digest=self.candidate_quiescence_digest,
            raw_crossover_digest=self.raw_crossover_digest,
            resident_policy=self.resident_policy,
            rates=rates,
            started_monotonic_s=self.started_monotonic_s,
            completed_monotonic_s=self.completed_monotonic_s,
        )
        if expected != self.evidence_digest:
            raise QualificationRunnerError("resident speed witness digest does not recompute")

    @classmethod
    def from_evidence(
        cls,
        value: ResidentCrossoverEvidence,
        plan: ResidentCrossoverPlan,
    ) -> "ResidentSpeedWitness":
        if (
            type(value) is not ResidentCrossoverEvidence
            or type(plan) is not ResidentCrossoverPlan
        ):
            raise QualificationRunnerError("resident crossover is not a speed witness")
        try:
            value.regrade(plan)
        except CrossoverRuntimeError as exc:
            raise QualificationRunnerError(str(exc)) from None
        kwargs = {
            "selected_delta_digest": value.selected_delta_digest,
            "candidate_launch_digest": value.candidate_execution.launch_digest,
            "calibration_digest": value.policy.calibration_digest,
            "calibration_context_digest": value.policy.calibration_context_digest,
            "workload_digest": value.workload_digest,
            "baseline_runtime_resource_policy_digest": (
                value.baseline_execution.resource_policy_digest
            ),
            "candidate_runtime_resource_policy_digest": (
                value.candidate_execution.resource_policy_digest
            ),
            "plan_digest": value.plan_digest,
            "baseline_lane_digest": value.baseline_lane_digest,
            "candidate_lane_digest": value.candidate_lane_digest,
            "baseline_quiescence_digest": value.baseline_quiescence.digest,
            "candidate_quiescence_digest": value.candidate_quiescence.digest,
            "raw_crossover_digest": value.digest,
            "resident_policy": value.policy,
            "rates": value.rates,
            "started_monotonic_s": value.started_monotonic_s,
            "completed_monotonic_s": value.completed_monotonic_s,
        }
        return cls(
            **kwargs,
            evidence_digest=_resident_speed_projection_digest(**kwargs),
        )

    @property
    def policy(self) -> SpeedEvidencePolicy:
        return SpeedEvidencePolicy.resident()

    @property
    def has_repeat(self) -> bool:
        return len(self.rates) == 5

    def regrade(
        self,
        calibration: CalibrationManifest,
        context: CalibrationContext,
        *,
        expected_policy: SpeedEvidencePolicy | None = None,
    ) -> tuple[QualificationDecision, str]:
        if expected_policy is not None and expected_policy != self.policy:
            raise QualificationRunnerError("resident speed witness policy differs")
        try:
            expected_resident = ResidentSpeedPolicy.from_calibration(
                max_stage_seconds=self.resident_policy.max_stage_seconds,
                max_qualification_seconds=(
                    self.resident_policy.max_qualification_seconds
                ),
                calibration=calibration,
                context=context,
            )
        except CrossoverRuntimeError as exc:
            raise QualificationRunnerError(str(exc)) from None
        if (
            self.resident_policy != expected_resident
            or self.calibration_digest != calibration.digest
            or self.calibration_context_digest != context.digest
            or self.workload_digest != context.workload_digest
        ):
            raise QualificationRunnerError("resident speed calibration authority differs")
        baselines = [
            row.tokens_per_second for row in self.rates if row.role.startswith("B")
        ]
        candidates = [
            row.tokens_per_second for row in self.rates if row.role.startswith("C")
        ]
        initial = score_speedup(
            baselines[:2],
            candidates[:1],
            min_margin=self.resident_policy.min_margin,
            k=self.resident_policy.noise_multiplier,
            max_noise=self.resident_policy.max_noise,
        )
        clear = (
            initial.confident
            and (
                initial.speedup <= initial.required - self.resident_policy.min_margin
                or initial.speedup >= initial.required + self.resident_policy.min_margin
            )
        )
        if (len(self.rates) == 3) != clear:
            raise QualificationRunnerError("resident adaptive read shape does not regrade")
        verdict = (
            initial
            if clear
            else score_speedup(
                baselines,
                candidates,
                min_margin=self.resident_policy.min_margin,
                k=self.resident_policy.noise_multiplier,
                max_noise=self.resident_policy.max_noise,
            )
        )
        grade = (
            QualificationDecision.NO_DECISION
            if not verdict.confident
            else QualificationDecision.PASS
            if verdict.passed_speedup
            else QualificationDecision.FAIL
        )
        return grade, format(verdict.speedup, ".17g")

    def to_dict(self) -> dict[str, object]:
        return {
            **{
                field: (
                    format(getattr(self, field), ".17g")
                    if field in {
                        "started_monotonic_s",
                        "completed_monotonic_s",
                    }
                    else getattr(self, field)
                )
                for field in self.__dataclass_fields__
                if field not in {"resident_policy", "rates"}
            },
            "rates": [row.to_dict() for row in self.rates],
            "resident_policy": self.resident_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "ResidentSpeedWitness":
        raw = _strict(value, set(cls.__dataclass_fields__), "resident speed witness")
        if type(raw["rates"]) is not list:
            raise QualificationRunnerError("resident speed witness rates are malformed")
        return cls(
            **{
                **raw,
                "resident_policy": ResidentSpeedPolicy.from_dict(
                    raw["resident_policy"]
                ),
                "rates": tuple(
                    ResidentReadRate.from_dict(row) for row in raw["rates"]
                ),
                "started_monotonic_s": float(raw["started_monotonic_s"]),
                "completed_monotonic_s": float(raw["completed_monotonic_s"]),
            }
        )  # type: ignore[arg-type]


@dataclass(frozen=True)
class AuditWitness:
    """Raw untimed eager slot-audit facts regraded only by the trusted host."""

    selected_delta_digest: str
    candidate_launch_digest: str
    execution_identity_digest: str
    session_id: str
    runtime_resource_policy_digest: str
    policy: SlotAuditPolicy
    receipts: tuple[AuditReceiptFacts, ...]
    decision: QualificationDecision
    detail: str

    def __post_init__(self) -> None:
        for name in (
            "selected_delta_digest",
            "candidate_launch_digest",
            "execution_identity_digest",
            "runtime_resource_policy_digest",
        ):
            object.__setattr__(
                self, name, require_sha256_hex(getattr(self, name), field=name)
            )
        if (
            not isinstance(self.session_id, str)
            or len(self.session_id) != 32
            or any(char not in "0123456789abcdef" for char in self.session_id)
            or self.session_id == "0" * 32
            or type(self.policy) is not SlotAuditPolicy
        ):
            raise QualificationRunnerError("audit witness identity is malformed")
        receipts = tuple(self.receipts)
        if any(type(row) is not AuditReceiptFacts for row in receipts):
            raise QualificationRunnerError("audit witness receipts are not typed")
        object.__setattr__(self, "receipts", receipts)
        object.__setattr__(self, "decision", _decision(self.decision))
        passed, detail = self.regrade()
        expected = (
            QualificationDecision.PASS if passed else QualificationDecision.FAIL
        )
        if (
            self.decision is not expected
            or not isinstance(self.detail, str)
            or self.detail != detail
        ):
            raise QualificationRunnerError(
                "audit witness verdict was not independently host-regraded"
            )

    def regrade(self) -> tuple[bool, str]:
        from optima.audit_gate import gate

        return gate(
            [row.to_gate_dict() for row in self.receipts],
            min_calls=self.policy.minimum_calls,
            expected_slots=self.policy.expected_slots,
            expected_member_count=self.policy.expected_member_count,
        )

    @classmethod
    def from_execution(
        cls,
        execution: EngineExecutionEvidence,
        *,
        selected_delta_digest: str,
        policy: SlotAuditPolicy,
    ) -> "AuditWitness":
        if type(execution) is not EngineExecutionEvidence:
            raise QualificationRunnerError("audit execution evidence is not typed")
        session = execution.session
        if (
            session.audit_policy_digest != policy.digest
            or not session.audit_receipts
        ):
            raise QualificationRunnerError(
                "audit session lacks policy-bound raw receipt evidence"
            )
        execution_identity = canonical_digest(
            "optima.qualification.slot-audit-execution.v1",
            {
                "arena_model_receipt": execution.arena_model_receipt_digest,
                "launch": execution.launch_digest,
                "native_publication": execution.native_publication_digest,
                "prebuild": execution.prebuild.build_spec_digest,
                "recovered_leases": list(execution.recovered_lease_ids),
                "runtime_argv": execution.runtime_argv_sha256,
                "runtime_preflight": execution.runtime_preflight_receipt_sha256,
                "runtime_resource_policy": execution.resource_policy_digest,
                "session_id": session.session_id,
                "audit_policy": policy.digest,
                # The isolated-session protocol carries bounded JSON numbers for
                # these raw runtime facts.  Authority identities do not: stack
                # canonical JSON deliberately rejects floats.  Encode through
                # the same lossless ``.17g`` projection used by the durable
                # witness before hashing, so the live witness and its reopened
                # record name exactly the same receipt semantics.
                "receipts": [_record_dict(row) for row in session.audit_receipts],
            },
        )
        from optima.audit_gate import gate

        passed, detail = gate(
            [row.to_gate_dict() for row in session.audit_receipts],
            min_calls=policy.minimum_calls,
            expected_slots=policy.expected_slots,
            expected_member_count=policy.expected_member_count,
        )
        return cls(
            selected_delta_digest,
            execution.launch_digest,
            execution_identity,
            session.session_id,
            execution.resource_policy_digest,
            policy,
            session.audit_receipts,
            QualificationDecision.PASS if passed else QualificationDecision.FAIL,
            detail,
        )

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: object) -> "AuditWitness":
        raw = _strict(value, set(cls.__dataclass_fields__), "audit witness")
        raw_receipts = raw["receipts"]
        if not isinstance(raw_receipts, list):
            raise QualificationRunnerError("audit witness receipts must be an array")
        receipts: list[AuditReceiptFacts] = []
        for value in raw_receipts:
            if not isinstance(value, dict):
                raise QualificationRunnerError("audit witness receipt is not an object")
            row = dict(value)
            for name, optional in (("worst_frac", False), ("min_ratio", True)):
                encoded = row.get(name)
                if optional and encoded is None:
                    continue
                if not isinstance(encoded, str):
                    raise QualificationRunnerError(
                        f"audit witness {name} is not a canonical decimal string"
                    )
                try:
                    decoded = float(encoded)
                except ValueError as exc:
                    raise QualificationRunnerError(
                        f"audit witness {name} is malformed"
                    ) from exc
                if not math.isfinite(decoded) or format(decoded, ".17g") != encoded:
                    raise QualificationRunnerError(
                        f"audit witness {name} is not canonical"
                    )
                row[name] = decoded
            receipts.append(AuditReceiptFacts.from_dict(row))
        return cls(
            **{
                **raw,
                "policy": SlotAuditPolicy.from_dict(raw["policy"]),
                "receipts": tuple(receipts),
            }
        )  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.qualification.slot-audit-witness.v1", self.to_dict())


@dataclass(frozen=True)
class QualificationStageExit:
    """Durable early terminal result; later expensive stages were not executed."""

    authority_digest: str
    source_digest: str
    selected_delta_digest: str
    stage: str
    decision: QualificationDecision
    reason: str
    speed_witness: ResidentSpeedWitness
    audit_witness: AuditWitness | None
    audit_started_monotonic_s: float | None
    audit_completed_monotonic_s: float | None
    terminal_quiescence_digest: str | None

    def __post_init__(self) -> None:
        for name in ("authority_digest", "source_digest", "selected_delta_digest"):
            object.__setattr__(
                self, name, require_sha256_hex(getattr(self, name), field=name)
            )
        object.__setattr__(self, "decision", _decision(self.decision))
        if (
            self.stage not in {"speed", "audit"}
            or type(self.speed_witness) is not ResidentSpeedWitness
            or self.speed_witness.selected_delta_digest
            != self.selected_delta_digest
        ):
            raise QualificationRunnerError("qualification stage exit is malformed")
        if self.terminal_quiescence_digest is not None:
            try:
                terminal_quiescence = require_sha256_hex(
                    self.terminal_quiescence_digest,
                    field="terminal_quiescence_digest",
                )
            except ValueError as exc:
                raise QualificationRunnerError(str(exc)) from None
            object.__setattr__(
                self, "terminal_quiescence_digest", terminal_quiescence
            )
        expected_reason = {
            ("speed", QualificationDecision.FAIL): "speed_regression",
            ("speed", QualificationDecision.NO_DECISION): "speed_noise",
            ("audit", QualificationDecision.FAIL): "slot_audit_failed",
        }.get((self.stage, self.decision))
        if self.reason != expected_reason:
            raise QualificationRunnerError("qualification stage-exit reason differs")
        if self.stage == "speed":
            if any(
                value is not None
                for value in (
                    self.audit_witness,
                    self.audit_started_monotonic_s,
                    self.audit_completed_monotonic_s,
                    self.terminal_quiescence_digest,
                )
            ):
                raise QualificationRunnerError("speed exit contains a later-stage witness")
        else:
            if (
                type(self.audit_witness) is not AuditWitness
                or self.audit_witness.decision is not QualificationDecision.FAIL
                or type(self.audit_started_monotonic_s) is not float
                or type(self.audit_completed_monotonic_s) is not float
                or not math.isfinite(self.audit_started_monotonic_s)
                or not math.isfinite(self.audit_completed_monotonic_s)
                or self.audit_started_monotonic_s
                < self.speed_witness.completed_monotonic_s
                or self.audit_completed_monotonic_s
                <= self.audit_started_monotonic_s
                or self.audit_completed_monotonic_s
                - self.speed_witness.started_monotonic_s
                > self.speed_witness.resident_policy.max_qualification_seconds
                or self.terminal_quiescence_digest is None
            ):
                raise QualificationRunnerError("audit stage-exit timing is malformed")

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_completed_monotonic_s": (
                None
                if self.audit_completed_monotonic_s is None
                else format(self.audit_completed_monotonic_s, ".17g")
            ),
            "audit_started_monotonic_s": (
                None
                if self.audit_started_monotonic_s is None
                else format(self.audit_started_monotonic_s, ".17g")
            ),
            "audit_witness": (
                None if self.audit_witness is None else self.audit_witness.to_dict()
            ),
            "authority_digest": self.authority_digest,
            "decision": self.decision.value,
            "reason": self.reason,
            "selected_delta_digest": self.selected_delta_digest,
            "source_digest": self.source_digest,
            "speed_witness": self.speed_witness.to_dict(),
            "stage": self.stage,
            "terminal_quiescence_digest": self.terminal_quiescence_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "QualificationStageExit":
        raw = _strict(value, set(cls.__dataclass_fields__), "qualification stage exit")

        def optional_time(name: str) -> float | None:
            encoded = raw[name]
            if encoded is None:
                return None
            try:
                result = float(encoded)
            except (TypeError, ValueError) as exc:
                raise QualificationRunnerError(
                    f"qualification stage-exit {name} is malformed"
                ) from exc
            if not math.isfinite(result) or format(result, ".17g") != encoded:
                raise QualificationRunnerError(
                    f"qualification stage-exit {name} is noncanonical"
                )
            return result

        return cls(
            **{
                **raw,
                "speed_witness": ResidentSpeedWitness.from_dict(
                    raw["speed_witness"]
                ),
                "audit_witness": (
                    None
                    if raw["audit_witness"] is None
                    else AuditWitness.from_dict(raw["audit_witness"])
                ),
                "audit_started_monotonic_s": optional_time(
                    "audit_started_monotonic_s"
                ),
                "audit_completed_monotonic_s": optional_time(
                    "audit_completed_monotonic_s"
                ),
            }
        )  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.qualification.stage-exit.v1", self.to_dict()
        )


@dataclass(frozen=True)
class QualificationTimingWitness:
    """Bounded speed/audit/T wall times for the production resident path."""

    policy_digest: str
    speed_evidence_digest: str
    audit_evidence_digest: str
    reference_session_digest: str
    max_qualification_seconds: int
    speed_started_monotonic_s: float
    speed_completed_monotonic_s: float
    audit_started_monotonic_s: float
    audit_completed_monotonic_s: float
    t_started_monotonic_s: float
    t_completed_monotonic_s: float
    qualification_completed_monotonic_s: float

    def __post_init__(self) -> None:
        for name in (
            "policy_digest",
            "speed_evidence_digest",
            "audit_evidence_digest",
            "reference_session_digest",
        ):
            object.__setattr__(
                self, name, require_sha256_hex(getattr(self, name), field=name)
            )
        if (
            type(self.max_qualification_seconds) is not int
            or not 60 <= self.max_qualification_seconds <= 14_400
        ):
            raise QualificationRunnerError(
                "qualification timing wall budget is malformed"
            )
        timestamps = tuple(
            getattr(self, name)
            for name in self.__dataclass_fields__
            if name.endswith("_monotonic_s")
        )
        if (
            any(type(row) is not float or not math.isfinite(row) for row in timestamps)
            or not (
                self.speed_started_monotonic_s
                < self.speed_completed_monotonic_s
                <= self.audit_started_monotonic_s
                < self.audit_completed_monotonic_s
                <= self.t_started_monotonic_s
                < self.t_completed_monotonic_s
                <= self.qualification_completed_monotonic_s
            )
            or self.qualification_completed_monotonic_s
            - self.speed_started_monotonic_s
            > self.max_qualification_seconds
        ):
            raise QualificationRunnerError(
                "qualification timing order or total wall time is invalid"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            name: (
                format(value, ".17g")
                if name.endswith("_monotonic_s")
                else value
            )
            for name, value in (
                (field, getattr(self, field))
                for field in self.__dataclass_fields__
            )
        }

    @classmethod
    def from_dict(cls, value: object) -> "QualificationTimingWitness":
        raw = _strict(value, set(cls.__dataclass_fields__), "qualification timing")
        for name in cls.__dataclass_fields__:
            if name.endswith("_monotonic_s"):
                encoded = raw[name]
                try:
                    decoded = float(encoded)
                except (TypeError, ValueError) as exc:
                    raise QualificationRunnerError(
                        f"qualification timing {name} is malformed"
                    ) from exc
                if not math.isfinite(decoded) or format(decoded, ".17g") != encoded:
                    raise QualificationRunnerError(
                        f"qualification timing {name} is noncanonical"
                    )
                raw[name] = decoded
        return cls(**raw)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.qualification.operational-timing.v1", self.to_dict()
        )


@dataclass(frozen=True)
class RepeatQualityWitness:
    """The independently teacher-graded B-prime/C-prime/B-double-prime leg."""

    quality_evidence_digest: str
    quality_decision: QualificationDecision
    candidate_mean_teacher_nll: str
    raw_quality_artifact: EvidenceArtifactRef
    raw_quality_binding: ReferenceQualityRawBinding
    t_request_sha256: str

    def __post_init__(self) -> None:
        for name in ("quality_evidence_digest", "t_request_sha256"):
            object.__setattr__(
                self, name, require_sha256_hex(getattr(self, name), field=name)
            )
        object.__setattr__(self, "quality_decision", _decision(self.quality_decision))
        if (
            type(self.raw_quality_artifact) is not EvidenceArtifactRef
            or type(self.raw_quality_binding) is not ReferenceQualityRawBinding
        ):
            raise QualificationRunnerError("repeat quality witness is not typed")
        try:
            nll = float(self.candidate_mean_teacher_nll)
        except (TypeError, ValueError) as exc:
            raise QualificationRunnerError("repeat candidate mean NLL is not numeric") from exc
        if not math.isfinite(nll) or nll < 0:
            raise QualificationRunnerError("repeat candidate mean NLL is nonfinite or negative")

    def to_dict(self) -> dict[str, object]:
        return _record_dict(self)

    @classmethod
    def from_dict(cls, value: object) -> "RepeatQualityWitness":
        raw = _strict(value, set(cls.__dataclass_fields__), "repeat quality witness")
        return cls(**{
            **raw,
            "raw_quality_artifact": EvidenceArtifactRef.from_dict(raw["raw_quality_artifact"]),
            "raw_quality_binding": ReferenceQualityRawBinding.from_dict(raw["raw_quality_binding"]),
        })  # type: ignore[arg-type]


def _quality_decision_pair(
    primary: QualificationDecision, repeat: QualificationDecision | None
) -> QualificationDecision:
    values = (primary,) if repeat is None else (primary, repeat)
    if QualificationDecision.FAIL in values:
        return QualificationDecision.FAIL
    if QualificationDecision.NO_DECISION in values:
        return QualificationDecision.NO_DECISION
    return QualificationDecision.PASS


def _speed_has_repeat(value: object) -> bool:
    if type(value) is SpeedWitness:
        return value.policy.version == 2
    if type(value) is ResidentSpeedWitness:
        return value.has_repeat
    raise QualificationRunnerError("speed witness type is unsupported")


def _candidate_runtime_resource_policy_digest(value: object) -> str:
    if type(value) is SpeedWitness:
        return value.runtime_resource_policy_digest
    if type(value) is ResidentSpeedWitness:
        return value.candidate_runtime_resource_policy_digest
    raise QualificationRunnerError("speed witness type is unsupported")


def _report_fields(value: object, *, include_repeat: bool) -> dict[str, object]:
    result = {
        row.name: _encode_record(getattr(value, row.name))
        for row in fields(value)
        if row.name != "repeat_quality"
    }
    if include_repeat:
        result["repeat_quality"] = _encode_record(getattr(value, "repeat_quality"))
    return result

@dataclass(frozen=True)
class CandidateQualificationReport:
    _domain: ClassVar[str] = "optima.qualification.candidate-report.v2"
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
    speed_witness: SpeedWitness | ResidentSpeedWitness
    t_request_sha256: str
    audit_evidence_digest: str
    audit_decision: QualificationDecision
    audit_witness: AuditWitness
    decision: QualificationDecision
    reason: str
    retryable: bool
    repeat_quality: RepeatQualityWitness | None = None

    def __post_init__(self) -> None:
        for field in (
            "selected_delta_digest", "marginal_arm_digest", "candidate_launch_digest",
            "profile_digest", "calibration_digest", "graph_grade_digest",
            "speed_evidence_digest", "quality_evidence_digest", "t_request_sha256",
            "audit_evidence_digest",
        ):
            object.__setattr__(self, field, require_sha256_hex(getattr(self, field), field=field))
        for field in (
            "graph_decision", "speed_decision", "quality_decision",
            "audit_decision", "decision",
        ):
            object.__setattr__(self, field, _decision(getattr(self, field)))
        if (
            type(self.raw_quality_artifact) is not EvidenceArtifactRef
            or type(self.raw_quality_binding) is not ReferenceQualityRawBinding
            or type(self.speed_witness) not in {SpeedWitness, ResidentSpeedWitness}
            or type(self.audit_witness) is not AuditWitness
        ):
            raise QualificationRunnerError("candidate evidence witness is not typed")
        if _speed_has_repeat(self.speed_witness) != (
            type(self.repeat_quality) is RepeatQualityWitness
        ):
            raise QualificationRunnerError(
                "candidate repeat quality coverage differs from speed policy"
            )
        expected = _aggregate_decision(
            self.graph_decision, self.speed_decision, self.quality_decision,
            self.audit_decision,
        )
        if (
            self.audit_witness.digest != self.audit_evidence_digest
            or self.audit_witness.decision is not self.audit_decision
            or self.audit_witness.selected_delta_digest != self.selected_delta_digest
            or self.audit_witness.candidate_launch_digest != self.candidate_launch_digest
            or self.audit_witness.runtime_resource_policy_digest
            != _candidate_runtime_resource_policy_digest(self.speed_witness)
            or self.decision is not expected
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
        return _report_fields(self, include_repeat=self.repeat_quality is not None)

    @classmethod
    def from_dict(cls, value: object) -> "CandidateQualificationReport":
        fields_v2 = set(cls.__dataclass_fields__) - {"_domain"}
        fields_v1 = fields_v2 - {"repeat_quality"}
        if type(value) is not dict:
            raise QualificationRunnerError("candidate report is not an object")
        raw = _strict(
            value,
            fields_v2 if "repeat_quality" in value else fields_v1,
            "candidate report",
        )
        return cls(**{
            **raw,
            "raw_quality_artifact": EvidenceArtifactRef.from_dict(raw["raw_quality_artifact"]),
            "raw_quality_binding": ReferenceQualityRawBinding.from_dict(raw["raw_quality_binding"]),
            "speed_witness": (
                ResidentSpeedWitness.from_dict(raw["speed_witness"])
                if type(raw["speed_witness"]) is dict
                and "resident_policy" in raw["speed_witness"]
                else SpeedWitness.from_dict(raw["speed_witness"])
            ),
            "audit_witness": AuditWitness.from_dict(raw["audit_witness"]),
            "repeat_quality": (
                RepeatQualityWitness.from_dict(raw["repeat_quality"])
                if "repeat_quality" in raw
                else None
            ),
        })  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        domain = self._domain if self.repeat_quality is None else f"{self._domain}.repeat"
        return canonical_digest(domain, self.to_dict())


@dataclass(frozen=True)
class DiscoveryCandidateQualificationReport:
    _domain: ClassVar[str] = "optima.qualification.discovery-candidate-report.v2"
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
    speed_witness: SpeedWitness | ResidentSpeedWitness
    t_request_sha256: str
    audit_evidence_digest: str
    audit_decision: QualificationDecision
    audit_witness: AuditWitness
    decision: QualificationDecision
    reason: str
    retryable: bool
    repeat_quality: RepeatQualityWitness | None = None

    def __post_init__(self) -> None:
        for field in (
            "selected_delta_digest", "discovery_arm_digest", "proposal_digest",
            "candidate_launch_digest", "profile_digest", "calibration_digest",
            "speed_evidence_digest",
            "quality_evidence_digest", "t_request_sha256",
            "audit_evidence_digest",
        ):
            object.__setattr__(
                self, field, require_sha256_hex(getattr(self, field), field=field)
            )
        for field in (
            "speed_decision", "quality_decision", "audit_decision", "decision"
        ):
            object.__setattr__(self, field, _decision(getattr(self, field)))
        if (
            type(self.execution_grade) is not DiscoveryExecutionGrade
            or type(self.raw_quality_artifact) is not EvidenceArtifactRef
            or type(self.raw_quality_binding) is not ReferenceQualityRawBinding
            or type(self.speed_witness) not in {SpeedWitness, ResidentSpeedWitness}
            or type(self.audit_witness) is not AuditWitness
        ):
            raise QualificationRunnerError("discovery evidence witness is not typed")
        if _speed_has_repeat(self.speed_witness) != (
            type(self.repeat_quality) is RepeatQualityWitness
        ):
            raise QualificationRunnerError(
                "discovery repeat quality coverage differs from speed policy"
            )
        expected = _aggregate_decision(
            self.execution_grade.decision, self.speed_decision,
            self.quality_decision, self.audit_decision,
        )
        if (
            self.audit_witness.digest != self.audit_evidence_digest
            or self.audit_witness.decision is not self.audit_decision
            or self.audit_witness.selected_delta_digest != self.selected_delta_digest
            or self.audit_witness.candidate_launch_digest != self.candidate_launch_digest
            or self.audit_witness.runtime_resource_policy_digest
            != _candidate_runtime_resource_policy_digest(self.speed_witness)
            or self.decision is not expected
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
        return _report_fields(self, include_repeat=self.repeat_quality is not None)

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryCandidateQualificationReport":
        fields_v2 = set(cls.__dataclass_fields__) - {"_domain"}
        fields_v1 = fields_v2 - {"repeat_quality"}
        if type(value) is not dict:
            raise QualificationRunnerError("discovery candidate report is not an object")
        raw = _strict(
            value,
            fields_v2 if "repeat_quality" in value else fields_v1,
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
            "speed_witness": (
                ResidentSpeedWitness.from_dict(raw["speed_witness"])
                if type(raw["speed_witness"]) is dict
                and "resident_policy" in raw["speed_witness"]
                else SpeedWitness.from_dict(raw["speed_witness"])
            ),
            "audit_witness": AuditWitness.from_dict(raw["audit_witness"]),
            "repeat_quality": (
                RepeatQualityWitness.from_dict(raw["repeat_quality"])
                if "repeat_quality" in raw
                else None
            ),
        })  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        domain = self._domain if self.repeat_quality is None else f"{self._domain}.repeat"
        return canonical_digest(domain, self.to_dict())


def _attempt_speed_policy(reports: tuple[object, ...]) -> SpeedEvidencePolicy:
    policies = tuple(
        row.speed_witness.policy
        for row in reports
        if type(row) in {
            CandidateQualificationReport,
            DiscoveryCandidateQualificationReport,
        }
    )
    if len(policies) != len(reports) or not policies or any(
        row != policies[0] for row in policies
    ):
        raise QualificationRunnerError("qualification reports mix speed evidence policies")
    return policies[0]

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
    operational_timing: QualificationTimingWitness | None = None

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
        speed_policy = _attempt_speed_policy(reports)
        if (speed_policy.version == 3) != (
            type(self.operational_timing) is QualificationTimingWitness
        ):
            raise QualificationRunnerError(
                "resident attempt operational timing coverage differs"
            )
        if self.operational_timing is not None:
            timing = self.operational_timing
            report = reports[0] if len(reports) == 1 else None
            if (
                report is None
                or type(report.speed_witness) is not ResidentSpeedWitness
                or timing.policy_digest != report.speed_witness.resident_policy.digest
                or timing.speed_evidence_digest != report.speed_evidence_digest
                or timing.audit_evidence_digest != report.audit_evidence_digest
                or timing.reference_session_digest
                != self.reference_execution.session_digest
                or timing.max_qualification_seconds
                != report.speed_witness.resident_policy.max_qualification_seconds
                or timing.speed_started_monotonic_s
                != report.speed_witness.started_monotonic_s
                or timing.speed_completed_monotonic_s
                != report.speed_witness.completed_monotonic_s
                or timing.qualification_completed_monotonic_s
                > self.teardown_after_t.observed_monotonic_s
            ):
                raise QualificationRunnerError(
                    "resident attempt timing differs from retained evidence"
                )
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
        result = _record_dict(self)
        if self.operational_timing is None:
            result.pop("operational_timing")
        return result

    @classmethod
    def from_dict(cls, value: object) -> "CohortQualificationAttempt":
        fields_v3 = set(cls.__dataclass_fields__) - {"_domain"}
        fields_legacy = fields_v3 - {"operational_timing"}
        if type(value) is not dict:
            raise QualificationRunnerError("cohort attempt is not an object")
        raw = _strict(
            value,
            fields_v3 if "operational_timing" in value else fields_legacy,
            "cohort attempt",
        )
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
            "operational_timing": (
                QualificationTimingWitness.from_dict(raw["operational_timing"])
                if "operational_timing" in raw
                else None
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
        version = _attempt_speed_policy(self.reports).version
        domain = {
            1: self._domain,
            2: f"{self._domain}.repeat",
            3: f"{self._domain}.resident",
        }[version]
        return canonical_digest(domain, self.to_dict())


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
        _attempt_speed_policy(reports)
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
        domain = (
            self._domain
            if _attempt_speed_policy(self.reports).version == 1
            else f"{self._domain}.repeat"
        )
        return canonical_digest(domain, self.to_dict())


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
    *,
    candidate_read: int = 1,
) -> tuple[tuple[str, str, tuple[dict[str, object], ...]], ...]:
    from optima.eval.qualification import _quality_leg_lifecycle

    lifecycle = _quality_leg_lifecycle(lifecycle, candidate_read)
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
    candidate_read: int = 1,
) -> ReferenceRequest:
    prompts = []
    for prompt_digest, prompt_text, frames in _selected_frames(
        lifecycle,
        authority.selected_delta_digest,
        selection.selected_prompt_digests,
        candidate_read=candidate_read,
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
    *,
    candidate_read: int = 1,
) -> ReferenceQualityRawArtifact:
    profile = authority.profile
    request = exchange.request
    request_digest = exchange.request_sha256
    frames = _selected_frames(
        lifecycle,
        authority.selected_delta_digest,
        selection.selected_prompt_digests,
        candidate_read=candidate_read,
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
            candidate_read=candidate_read,
        ),
        selected_trajectory_projection_digest(
            lifecycle,
            selected_delta_digest=authority.selected_delta_digest,
            selected_prompt_digests=selection.selected_prompt_digests,
            candidate_read=candidate_read,
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


def _run_slot_audits(
    value: CausalQualificationInput,
    lifecycle: MarginalLifecycleEvidence | ResidentMarginalLifecycleEvidence,
    *,
    executor: OCIEngineExecutor,
    deadline: float,
) -> tuple[dict[str, AuditWitness], float]:
    """Run one independent eager, untimed candidate role per sealed C arm."""

    if type(lifecycle) is ResidentMarginalLifecycleEvidence:
        timed_session_ids = set(lifecycle.timed_session_ids)
        if type(value.resident_audit_plan) is not SessionExecutionPlan:
            raise QualificationRunnerError(
                "resident audit session lacks its sealed audit-only plan"
            )
        audit_plans = (value.resident_audit_plan,)
    else:
        timed_session_ids = {
            lifecycle.baseline_before.session.session_id,
            lifecycle.baseline_after.session.session_id,
            *(row.execution.session.session_id for row in lifecycle.candidates),
            *(
                row.execution.session.session_id
                for row in lifecycle.candidates_repeat
            ),
            *(
                (lifecycle.baseline_third.session.session_id,)
                if lifecycle.baseline_third is not None
                else ()
            ),
        }
        audit_plans = tuple(
            replace(prepared.session_plan, audit_policy=policy)
            for prepared, policy in zip(
                value.prepared.candidates,
                value.audit_policies,
                strict=True,
            )
        )
    witnesses: dict[str, AuditWitness] = {}
    last_completed = 0.0
    for prepared, authority, policy, audit_plan in zip(
        value.prepared.candidates,
        value.candidates,
        value.audit_policies,
        audit_plans,
        strict=True,
    ):
        if audit_plan.audit_policy != policy:
            raise QualificationRunnerError(
                "slot audit execution plan differs from its sealed policy"
            )
        execution = executor.execute(
            prepared.launch,
            prepared.binding.launch_binding,
            value.model_mount,
            audit_plan,
            deadline=deadline,
        )
        if (
            execution.session.session_id in timed_session_ids
            or execution.resource_policy_digest
            != value.expected_runtime_resource_policy_digest
        ):
            raise QualificationRunnerError(
                "slot audit execution reused a timed role or changed runtime policy"
            )
        witness = AuditWitness.from_execution(
            execution,
            selected_delta_digest=authority.selected_delta_digest,
            policy=policy,
        )
        witnesses[authority.selected_delta_digest] = witness
        last_completed = max(
            last_completed,
            execution.device_receipts[-1].completed_monotonic_s,
        )
    if len(witnesses) != len(value.candidates):
        raise QualificationRunnerError("slot audit cohort coverage is incomplete")
    return witnesses, last_completed

def _report_reason(
    graph: GraphVerificationGrade,
    speed: QualificationDecision,
    quality: ReferenceQualityVerdict,
    audit: AuditWitness,
    repeat_quality: ReferenceQualityVerdict | None = None,
) -> str:
    if graph.decision is not QualificationDecision.PASS:
        return graph.reason
    if audit.decision is not QualificationDecision.PASS:
        return "slot_audit_failed"
    if speed is QualificationDecision.NO_DECISION:
        return "speed_noise"
    if speed is QualificationDecision.FAIL:
        return "speed_regression"
    if quality.decision == "FAIL":
        return "quality_regression"
    if quality.decision == "NO_DECISION":
        return "quality_overlap"
    if repeat_quality is not None and repeat_quality.decision == "FAIL":
        return "quality_repeat_regression"
    if repeat_quality is not None and repeat_quality.decision == "NO_DECISION":
        return "quality_repeat_overlap"
    return "qualified"


def _discovery_report_reason(
    execution: DiscoveryExecutionGrade,
    speed: QualificationDecision,
    quality: ReferenceQualityVerdict,
    audit: AuditWitness,
    repeat_quality: ReferenceQualityVerdict | None = None,
) -> str:
    if execution.decision is not QualificationDecision.PASS:
        return execution.reason
    if audit.decision is not QualificationDecision.PASS:
        return "slot_audit_failed"
    if speed is QualificationDecision.NO_DECISION:
        return "speed_noise"
    if speed is QualificationDecision.FAIL:
        return "speed_regression"
    if quality.decision == "FAIL":
        return "quality_regression"
    if quality.decision == "NO_DECISION":
        return "quality_overlap"
    if repeat_quality is not None and repeat_quality.decision == "FAIL":
        return "quality_repeat_regression"
    if repeat_quality is not None and repeat_quality.decision == "NO_DECISION":
        return "quality_repeat_overlap"
    return "qualified"


def _audit_session_plan_digest(plan: SessionExecutionPlan) -> str:
    """Bind every host-owned input of one fresh, untimed audit session."""

    if type(plan) is not SessionExecutionPlan or plan.audit_policy is None:
        raise QualificationRunnerError("resident audit plan is not exact and armed")
    return canonical_digest(
        "optima.qualification.audit-session-plan.v1",
        {
            "audit_policy": plan.audit_policy.digest,
            "conditioning_count": plan.conditioning_count,
            "discovery_overlay_identity": (
                plan.expected_discovery_overlay_identity_digest
            ),
            "engine_config": plan.expected_engine_config_digest,
            "launch": plan.launch_digest,
            "max_new_tokens": plan.max_new_tokens,
            "preflight": plan.expected_preflight.digest,
            "prompt_batches": plan.prompt_batches,
            "temperature": format(plan.temperature, ".17g"),
            "top_logprobs_num": plan.top_logprobs_num,
            "warmup_count": plan.warmup_count,
        },
    )


def qualification_authority_digest(value: CausalQualificationInput) -> str:
    """Bind the durable attempt to all validator-owned inputs available before B."""

    if all(type(row) is CandidateQualificationAuthority for row in value.candidates):
        payload = {
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
        "slot_audit_policies": [row.to_dict() for row in value.audit_policies],
        "reference": [value.pristine_stack.digest, value.pristine_launch.digest,
                      value.pristine_binding.native_build_spec.digest,
                      value.pristine_binding.controller_distribution_digest,
                      value.pristine_binding.runtime_preflight_receipt.sha256,
                      value.reference_engine_config.digest, value.reference_preflight.digest],
        "source": value.prepared.source.digest,
        }
        if value.speed_evidence_policy.version == 1:
            return canonical_digest(
                "optima.qualification.causal-authority.audit-v1", payload
            )
        payload["speed_evidence_policy"] = value.speed_evidence_policy.to_dict()
        if value.speed_evidence_policy.version == 3:
            assert value.resident_speed_plan is not None
            assert value.resident_audit_plan is not None
            payload["resident_speed_plan"] = value.resident_speed_plan.digest
            payload["resident_audit_plan"] = _audit_session_plan_digest(
                value.resident_audit_plan
            )
            if (
                value.speed_stage_disposition
                is SpeedStageDisposition.CALIBRATION_OBSERVATION
            ):
                payload["speed_stage_disposition"] = (
                    value.speed_stage_disposition.value
                )
                return canonical_digest(
                    "optima.qualification.causal-authority.v3.audit-v1."
                    "calibration-observation-v1",
                    payload,
                )
        return canonical_digest(
            (
                "optima.qualification.causal-authority.v3.audit-v1"
                if value.speed_evidence_policy.version == 3
                else "optima.qualification.causal-authority.v2.audit-v1"
            ),
            payload,
        )
    if len(value.candidates) != 1 or type(
        value.candidates[0]
    ) is not DiscoveryCandidateQualificationAuthority:
        raise QualificationRunnerError("qualification authority mode is inconsistent")
    authority = value.candidates[0]
    prepared = value.prepared.candidates[0]
    payload = {
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
        "slot_audit_policies": [row.to_dict() for row in value.audit_policies],
        "reference": [value.pristine_stack.digest, value.pristine_launch.digest,
                      value.pristine_binding.native_build_spec.digest,
                      value.pristine_binding.controller_distribution_digest,
                      value.pristine_binding.runtime_preflight_receipt.sha256,
                      value.reference_engine_config.digest,
                      value.reference_preflight.digest],
        "source": value.prepared.source.digest,
    }
    if value.speed_evidence_policy.version == 1:
        return canonical_digest(
            "optima.qualification.discovery-causal-authority.audit-v1", payload
        )
    payload["speed_evidence_policy"] = value.speed_evidence_policy.to_dict()
    return canonical_digest(
        "optima.qualification.discovery-causal-authority.v2.audit-v1", payload
    )

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
    expected_requests = tuple(
        request
        for report in attempt.reports
        for request in (
            (report.t_request_sha256,)
            if report.repeat_quality is None
            else (
                report.t_request_sha256,
                report.repeat_quality.t_request_sha256,
            )
        )
    )
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
            expected_requests)
        or pre[2] != post[2] or pre[1] >= post[1] or pre[4] != post[4]
        or pre[5:7] != post[5:7] or pre[6] != expected.expected_device_policy_digest
        or len(pre[4]) != expected.pristine_launch.hardware.visible_gpu_count
        or float(pre[7]) < attempt.entropy_observed_monotonic_s
        or float(pre[8]) > float(post[7])
        or float(post[8]) > attempt.teardown_after_t.observed_monotonic_s
    ):
        raise QualificationRunnerError("pristine T execution witness differs from causal authority")

def publish_qualification_stage_exit(
    root: Path, result: QualificationStageExit
) -> EvidenceArtifactRef:
    if type(result) is not QualificationStageExit:
        raise QualificationRunnerError("qualification stage exit is not typed")
    return publish_evidence(
        root,
        canonical_json_bytes(result.to_dict()),
        domain=STAGE_EXIT_DOMAIN,
        media_type="application/json",
        schema=STAGE_EXIT_SCHEMA,
    )


def reopen_qualification_stage_exit(
    root: Path,
    reference: EvidenceArtifactRef,
    *,
    expected: CausalQualificationInput,
) -> QualificationStageExit:
    try:
        if (
            expected.speed_evidence_policy != SpeedEvidencePolicy.resident()
            or type(expected.resident_speed_plan) is not ResidentCrossoverPlan
            or len(expected.candidates) != 1
            or type(reference) is not EvidenceArtifactRef
            or (
                reference.domain,
                reference.media_type,
                reference.schema,
            )
            != (STAGE_EXIT_DOMAIN, "application/json", STAGE_EXIT_SCHEMA)
        ):
            raise QualificationRunnerError(
                "qualification stage-exit authority is unsupported"
            )
        payload = _canonical_payload(reopen_evidence(root, reference))
        result = QualificationStageExit.from_dict(payload)
        plan = expected.resident_speed_plan
        witness = result.speed_witness
        if (
            result.to_dict() != payload
            or result.authority_digest != qualification_authority_digest(expected)
            or result.source_digest != expected.prepared.source.digest
            or result.selected_delta_digest
            != expected.candidates[0].selected_delta_digest
            or witness.plan_digest != plan.digest
            or witness.candidate_launch_digest != plan.candidate.launch.digest
            or witness.baseline_runtime_resource_policy_digest
            != plan.baseline.runtime_resource_policy_digest
            or witness.candidate_runtime_resource_policy_digest
            != plan.candidate.runtime_resource_policy_digest
            or witness.candidate_runtime_resource_policy_digest
            != expected.expected_runtime_resource_policy_digest
            or witness.baseline_lane_digest != plan.baseline_lane_digest
            or witness.candidate_lane_digest != plan.candidate_lane_digest
            or any(
                row.lane_digest
                != (
                    plan.baseline_lane_digest
                    if row.role.startswith("B")
                    else plan.candidate_lane_digest
                )
                for row in witness.rates
            )
        ):
            raise QualificationRunnerError(
                "qualification stage exit differs from its authority"
            )
        calibration = reopen_calibration_evidence(
            root,
            expected.calibration_artifact_ref,
            expected_threshold_policy=expected.calibration_threshold_policy,
            expected_manifest=expected.calibration_manifest,
            expected_context=expected.calibration_context,
        )
        speed_grade, _speedup = witness.regrade(
            calibration,
            expected.calibration_context,
            expected_policy=expected.speed_evidence_policy,
        )
        if result.stage == "speed":
            if (
                expected.speed_stage_disposition
                is SpeedStageDisposition.CALIBRATION_OBSERVATION
                or speed_grade is not result.decision
            ):
                raise QualificationRunnerError(
                    "speed stage exit does not independently regrade"
                )
        else:
            audit = result.audit_witness
            policy = expected.audit_policies[0]
            if (
                (
                    speed_grade is not QualificationDecision.PASS
                    and expected.speed_stage_disposition
                    is not SpeedStageDisposition.CALIBRATION_OBSERVATION
                )
                or type(audit) is not AuditWitness
                or audit.policy != policy
                or audit.selected_delta_digest != result.selected_delta_digest
                or audit.candidate_launch_digest != plan.candidate.launch.digest
                or audit.runtime_resource_policy_digest
                != expected.expected_runtime_resource_policy_digest
                or audit.session_id
                in {row.session_id for row in witness.rates}
            ):
                raise QualificationRunnerError(
                    "audit stage exit does not independently regrade"
                )
        return result
    except QualificationRunnerError:
        raise
    except Exception as exc:
        raise QualificationRunnerError(
            f"qualification stage exit cannot reopen: {exc}"
        ) from None


def publish_causal_qualification(
    root: Path, attempt: QualificationAttempt
) -> EvidenceArtifactRef:
    policy = _attempt_speed_policy(attempt.reports) if type(attempt) in {
        CohortQualificationAttempt, DiscoveryQualificationAttempt
    } else None
    if type(attempt) is CohortQualificationAttempt:
        domain = ATTEMPT_DOMAIN
        schema = {
            1: ATTEMPT_SCHEMA,
            2: ATTEMPT_SCHEMA_V2,
            3: ATTEMPT_SCHEMA_V3,
        }[policy.version]
    elif type(attempt) is DiscoveryQualificationAttempt:
        domain = DISCOVERY_ATTEMPT_DOMAIN
        schema = (
            DISCOVERY_ATTEMPT_SCHEMA
            if policy.version == 1
            else DISCOVERY_ATTEMPT_SCHEMA_V2
        )
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
            (
                DISCOVERY_ATTEMPT_DOMAIN,
                "application/json",
                DISCOVERY_ATTEMPT_SCHEMA
                if expected.speed_evidence_policy.version == 1
                else DISCOVERY_ATTEMPT_SCHEMA_V2,
            )
            if discovery_mode
            else (
                ATTEMPT_DOMAIN,
                "application/json",
                {
                    1: ATTEMPT_SCHEMA,
                    2: ATTEMPT_SCHEMA_V2,
                    3: ATTEMPT_SCHEMA_V3,
                }[expected.speed_evidence_policy.version],
            )
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
        if _attempt_speed_policy(attempt.reports) != expected.speed_evidence_policy:
            raise QualificationRunnerError("qualification speed evidence policy differs")
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
        for report, authority, prepared, audit_policy in zip(
            attempt.reports,
            expected.candidates,
            expected.prepared.candidates,
            expected.audit_policies,
            strict=True,
        ):
            raw = report.raw_quality_binding
            quality = score_reference_quality(
                reopen_reference_quality_evidence(root, report.raw_quality_artifact,
                                                  expected_binding=raw),
                calibration=calibration, expected_context=expected.calibration_context,
            )
            repeat_quality = None
            repeat_raw = None
            if report.repeat_quality is not None:
                repeat_raw = report.repeat_quality.raw_quality_binding
                repeat_quality = score_reference_quality(
                    reopen_reference_quality_evidence(
                        root,
                        report.repeat_quality.raw_quality_artifact,
                        expected_binding=repeat_raw,
                    ),
                    calibration=calibration,
                    expected_context=expected.calibration_context,
                )
            speed = report.speed_witness
            rates = speed.rates
            if expected.speed_evidence_policy.version == 3:
                plan = expected.resident_speed_plan
                if (
                    type(speed) is not ResidentSpeedWitness
                    or type(plan) is not ResidentCrossoverPlan
                    or speed.plan_digest != plan.digest
                    or speed.baseline_lane_digest
                    == speed.candidate_lane_digest
                    or speed.baseline_runtime_resource_policy_digest
                    != plan.baseline.runtime_resource_policy_digest
                    or speed.candidate_runtime_resource_policy_digest
                    != plan.candidate.runtime_resource_policy_digest
                    or speed.candidate_runtime_resource_policy_digest
                    != expected.expected_runtime_resource_policy_digest
                ):
                    raise QualificationRunnerError(
                        "resident speed witness differs from its authority"
                    )
                expected_launches = tuple(
                    plan.baseline.launch.digest
                    if row.role.startswith("B")
                    else plan.candidate.launch.digest
                    for row in rates
                )
                expected_lanes = tuple(
                    plan.baseline_lane_digest
                    if row.role.startswith("B")
                    else plan.candidate_lane_digest
                    for row in rates
                )
                baseline_sessions = {
                    row.session_id for row in rates if row.role.startswith("B")
                }
                candidate_sessions = {
                    row.session_id for row in rates if row.role.startswith("C")
                }
                session_shape_valid = (
                    len(baseline_sessions) == 1
                    and len(candidate_sessions) == 1
                    and baseline_sessions.isdisjoint(candidate_sessions)
                    and tuple(row.lane_digest for row in rates)
                    == expected_lanes
                )
            else:
                if type(speed) is not SpeedWitness:
                    raise QualificationRunnerError(
                        "legacy speed witness changed type"
                    )
                if (
                    speed.runtime_resource_policy_digest
                    != expected.expected_runtime_resource_policy_digest
                ):
                    raise QualificationRunnerError(
                        "legacy speed witness changed runtime policy"
                    )
                expected_launches = (
                    expected.prepared.baseline_launch.digest,
                    prepared.launch.digest,
                    expected.prepared.baseline_launch.digest,
                )
                if expected.speed_evidence_policy.version == 2:
                    expected_launches += (
                        prepared.launch.digest,
                        expected.prepared.baseline_launch.digest,
                    )
                session_shape_valid = (
                    len({row.session_id for row in rates})
                    == len(expected_launches)
                )
            if (
                (speed.selected_delta_digest, speed.candidate_launch_digest,
                 speed.evidence_digest)
                != (authority.selected_delta_digest, prepared.launch.digest,
                    report.speed_evidence_digest)
                or speed.policy != expected.speed_evidence_policy
                or tuple(row.launch_digest for row in rates) != expected_launches
                or not session_shape_valid
            ):
                raise QualificationRunnerError("speed witness differs from its marginal arm")
            speed_grade, speedup = speed.regrade(
                calibration,
                expected.calibration_context,
                expected_policy=expected.speed_evidence_policy,
            )
            audit_witness = report.audit_witness
            audit_passed, audit_detail = audit_witness.regrade()
            audit_grade = (
                QualificationDecision.PASS
                if audit_passed
                else QualificationDecision.FAIL
            )
            if (
                audit_witness.policy != audit_policy
                or audit_witness.selected_delta_digest
                != authority.selected_delta_digest
                or audit_witness.candidate_launch_digest != prepared.launch.digest
                or audit_witness.runtime_resource_policy_digest
                != expected.expected_runtime_resource_policy_digest
                or audit_witness.session_id in {row.session_id for row in rates}
                or audit_witness.decision is not audit_grade
                or audit_witness.detail != audit_detail
                or audit_witness.digest != report.audit_evidence_digest
            ):
                raise QualificationRunnerError(
                    "slot audit witness differs from its candidate authority"
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
            quality_grade = _quality_decision_pair(
                _decision(quality.decision),
                None if repeat_quality is None else _decision(repeat_quality.decision),
            )
            candidate_mean_teacher_nll = max(
                (
                    quality.candidate_mean_teacher_nll,
                    *(
                        (repeat_quality.candidate_mean_teacher_nll,)
                        if repeat_quality is not None
                        else ()
                    ),
                ),
                key=float,
            )
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
                    graph.decision, speed_grade, quality_grade, audit_grade
                )
                headline = (
                    report.marginal_arm_digest, report.candidate_launch_digest,
                    report.target_id, report.profile_digest,
                    report.calibration_digest, report.graph_grade_digest,
                    report.graph_decision, report.speed_evidence_digest,
                    report.speed_decision, report.speedup,
                    report.quality_evidence_digest, report.quality_decision,
                    report.candidate_mean_teacher_nll,
                    report.audit_evidence_digest, report.audit_decision,
                    report.decision,
                    report.reason, report.retryable,
                )
                expected_headline = (
                    prepared.arm.digest, prepared.launch.digest,
                    prepared.arm.transition.target_id, authority.profile.digest,
                    calibration.digest, graph.digest, graph.decision,
                    report.speed_witness.evidence_digest, speed_grade, speedup,
                    quality.evidence_digest, quality_grade,
                    candidate_mean_teacher_nll,
                    audit_witness.digest, audit_grade, decision,
                    _report_reason(
                        graph, speed_grade, quality, audit_witness,
                        repeat_quality,
                    ),
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
                    execution.decision, speed_grade, quality_grade, audit_grade
                )
                headline = (
                    report.discovery_arm_digest, report.proposal_digest,
                    report.candidate_launch_digest, report.profile_digest,
                    report.calibration_digest, report.execution_grade,
                    report.speed_evidence_digest, report.speed_decision,
                    report.speedup, report.quality_evidence_digest,
                    report.quality_decision, report.candidate_mean_teacher_nll,
                    report.audit_evidence_digest, report.audit_decision,
                    report.decision, report.reason, report.retryable,
                )
                expected_headline = (
                    prepared.arm.digest,
                    authority.execution_requirement.proposal_digest,
                    prepared.launch.digest, authority.profile.digest,
                    calibration.digest, execution,
                    report.speed_witness.evidence_digest, speed_grade, speedup,
                    quality.evidence_digest, quality_grade,
                    candidate_mean_teacher_nll,
                    audit_witness.digest, audit_grade, decision,
                    _discovery_report_reason(
                        execution, speed_grade, quality, audit_witness,
                        repeat_quality,
                    ),
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
            repeat_matches = repeat_raw is None and repeat_quality is None
            if repeat_raw is not None and repeat_quality is not None:
                if report.repeat_quality is None:  # defensive against future report unions
                    raise QualificationRunnerError("repeat quality witness disappeared")
                repeat_common = {
                    "calibration_digest": calibration.digest,
                    "candidate_lifecycle_digest": repeat_raw.candidate_lifecycle_digest,
                    "profile_digest": authority.profile.digest,
                    "selected_delta_digest": authority.selected_delta_digest,
                    "selection_digest": attempt.selection.digest,
                    "t_request_sha256": report.repeat_quality.t_request_sha256,
                    "t_session_digest": attempt.reference_session_digest,
                }
                if type(authority) is CandidateQualificationAuthority:
                    repeat_identity = canonical_digest(
                        "optima.qualification.candidate-identity",
                        {
                            **repeat_common,
                            "graph_requirement_digest": authority.graph_requirement.digest,
                        },
                    )
                else:
                    if type(authority) is not DiscoveryCandidateQualificationAuthority:
                        raise QualificationRunnerError(
                            "repeat quality authority type differs"
                        )
                    repeat_identity = canonical_digest(
                        "optima.qualification.discovery-candidate-identity",
                        {
                            **repeat_common,
                            "execution_requirement_digest": (
                                authority.execution_requirement.digest
                            ),
                        },
                    )
                repeat_binding = (
                    repeat_raw.qualification_identity_digest,
                    repeat_raw.reference_manifest_digest,
                    repeat_raw.calibration_digest,
                    repeat_raw.selection_digest,
                    repeat_raw.selected_prompt_digests,
                    repeat_raw.t_session_digest,
                    repeat_raw.t_request_sha256,
                    repeat_raw.support_policy_digest,
                    repeat_raw.hidden_task_plan_digest,
                    repeat_raw.nll_tail_threshold,
                    repeat_raw.tokens_per_prompt,
                    repeat_raw.topk_width,
                    repeat_raw.hidden_tasks_per_prompt,
                )
                repeat_expected_binding = (
                    repeat_identity,
                    authority.profile.reference.digest,
                    calibration.digest,
                    attempt.selection.digest,
                    attempt.selection.selected_prompt_digests,
                    attempt.reference_session_digest,
                    report.repeat_quality.t_request_sha256,
                    authority.profile.support_policy_digest,
                    derived_hidden_task_plan_digest(
                        authority.profile, attempt.selection.selected_prompt_digests
                    ),
                    authority.profile.nll_tail_threshold,
                    authority.profile.tokens_per_prompt,
                    authority.profile.topk_width,
                    authority.profile.hidden_tasks_per_prompt,
                )
                repeat_matches = (
                    repeat_binding == repeat_expected_binding
                    and (
                        report.repeat_quality.quality_evidence_digest,
                        report.repeat_quality.quality_decision,
                        report.repeat_quality.candidate_mean_teacher_nll,
                    )
                    == (
                        repeat_quality.evidence_digest,
                        _decision(repeat_quality.decision),
                        repeat_quality.candidate_mean_teacher_nll,
                    )
                )
            if (
                binding != expected_binding
                or headline != expected_headline
                or not repeat_matches
            ):
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
    resident_baseline_executor: OCIEngineExecutor | None = None,
    entropy_provider: EntropyProvider,
    hidden_judge: HiddenJudge,
    deadline: float,
    id_factory: Callable[[], str] | None = None,
) -> EvidenceArtifactRef:
    """Run one complete causal cohort or raise without a partial PASS.

    The speed read shape comes only from ``value.speed_evidence_policy`` so an
    operator cannot silently change the estimator after sealing intake authority.
    """

    resident_mode = value.speed_evidence_policy.version == 3
    if (
        type(executor) is not OCIEngineExecutor
        or not callable(entropy_provider)
        or not callable(hidden_judge)
        or (
            resident_mode
            and (
                type(resident_baseline_executor) is not OCIEngineExecutor
                or resident_baseline_executor is executor
            )
        )
        or (not resident_mode and resident_baseline_executor is not None)
    ):
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
    if resident_mode:
        assert value.resident_speed_plan is not None
        observed_start = float(executor.manager.clock())
        deadline = min(
            float(deadline),
            observed_start
            + value.resident_speed_plan.policy.max_qualification_seconds,
        )
        if not math.isfinite(observed_start) or deadline <= observed_start:
            raise QualificationRunnerError(
                "resident qualification has no wall-clock budget"
            )
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

    resident_speed_witness: ResidentSpeedWitness | None = None
    if resident_mode:
        assert resident_baseline_executor is not None
        assert value.resident_speed_plan is not None
        try:
            crossover = run_resident_crossover_speed(
                value.resident_speed_plan,
                baseline_executor=resident_baseline_executor,
                candidate_executor=executor,
                model_mount=value.model_mount,
                deadline=float(deadline),
            )
            lifecycle: MarginalLifecycleEvidence | ResidentMarginalLifecycleEvidence = (
                ResidentMarginalLifecycleEvidence(
                    value.prepared,
                    value.resident_speed_plan,
                    crossover,
                )
            )
            resident_speed_witness = ResidentSpeedWitness.from_evidence(
                crossover, value.resident_speed_plan
            )
        except CrossoverRuntimeError as exc:
            raise QualificationRunnerError(str(exc)) from None
        speed_grade, _speedup = resident_speed_witness.regrade(
            calibration,
            value.calibration_context,
            expected_policy=value.speed_evidence_policy,
        )
        if (
            speed_grade is not QualificationDecision.PASS
            and value.speed_stage_disposition is SpeedStageDisposition.TERMINAL
        ):
            terminal = QualificationStageExit(
                qualification_authority_digest(value),
                value.prepared.source.digest,
                value.candidates[0].selected_delta_digest,
                "speed",
                speed_grade,
                (
                    "speed_noise"
                    if speed_grade is QualificationDecision.NO_DECISION
                    else "speed_regression"
                ),
                resident_speed_witness,
                None,
                None,
                None,
                None,
            )
            reference = publish_qualification_stage_exit(
                value.evidence_root, terminal
            )
            reopen_qualification_stage_exit(
                value.evidence_root, reference, expected=value
            )
            return reference
        quality_reads = 2 if crossover.escalated else 1
    else:
        quality_reads = value.speed_evidence_policy.candidate_reads

    with executor.exclusive_transaction():
        if not resident_mode:
            lifecycle = run_marginal_lifecycle(
                value.prepared,
                executor=executor,
                model_mount=value.model_mount,
                deadline=float(deadline),
                candidate_reads=value.speed_evidence_policy.candidate_reads,
            )
        audit_started = float(executor.manager.clock())
        audit_witnesses, audit_last_completed = _run_slot_audits(
            value,
            lifecycle,
            executor=executor,
            deadline=float(deadline),
        )
        audit_completed = float(executor.manager.clock())
        if resident_mode and any(
            row.decision is not QualificationDecision.PASS
            for row in audit_witnesses.values()
        ):
            teardown = executor.prove_quiescent()
            if teardown.observed_monotonic_s < audit_last_completed:
                raise QualificationRunnerError(
                    "audit-exit quiescence predates candidate teardown"
                )
            assert resident_speed_witness is not None
            audit = audit_witnesses[value.candidates[0].selected_delta_digest]
            terminal = QualificationStageExit(
                qualification_authority_digest(value),
                value.prepared.source.digest,
                value.candidates[0].selected_delta_digest,
                "audit",
                QualificationDecision.FAIL,
                "slot_audit_failed",
                resident_speed_witness,
                audit,
                audit_started,
                audit_completed,
                teardown.digest,
            )
            reference = publish_qualification_stage_exit(
                value.evidence_root, terminal
            )
            reopen_qualification_stage_exit(
                value.evidence_root, reference, expected=value
            )
            return reference
        discovery_grades: dict[str, DiscoveryExecutionGrade] = {}
        for authority in value.candidates:
            if type(authority) is DiscoveryCandidateQualificationAuthority:
                discovery_grades[authority.selected_delta_digest] = (
                    grade_discovery_execution(
                        authority.execution_requirement, lifecycle
                    )
                )
        teardown_before = executor.prove_quiescent()
        # Bind quiescence to the FINAL executed baseline (B'' under repeat reads,
        # B-prime otherwise) — baseline_after is mid-run in the 5-leg shape.
        last_post = max(
            lifecycle.final_baseline.device_receipts[-1].completed_monotonic_s,
            audit_last_completed,
        )
        if teardown_before.observed_monotonic_s < last_post:
            raise QualificationRunnerError("pre-T quiescence predates the final baseline teardown")
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
        request_plan_payload = {
                "candidate_deltas": [row.selected_delta_digest for row in value.candidates],
                "cohort_trajectory_digest": cohort_trajectory_digest(lifecycle),
                "reference_manifest_digest": value.candidates[0].profile.reference.digest,
                "selection_digest": selection.digest,
        }
        if value.speed_evidence_policy.version != 1:
            request_plan_payload["speed_evidence_policy"] = (
                value.speed_evidence_policy.to_dict()
            )
        if resident_speed_witness is not None:
            request_plan_payload["resident_speed_evidence"] = (
                resident_speed_witness.evidence_digest
            )
        request_plan_digest = canonical_digest(
            "optima.qualification.reference-request-plan",
            request_plan_payload,
        )
        session_id = make_id()
        request_rows: list[ReferenceRequest] = []
        for authority in value.candidates:
            for candidate_read in range(1, quality_reads + 1):
                kwargs = {
                    "session_id": session_id,
                    "plan_digest": request_plan_digest,
                    "request_id": make_id(),
                    "nonce": make_id(),
                    "index": len(request_rows),
                }
                if candidate_read == 1:
                    # Preserve historical call/serialization behavior exactly.
                    request = _reference_request(
                        lifecycle, authority, selection, **kwargs
                    )
                else:
                    request = _reference_request(
                        lifecycle,
                        authority,
                        selection,
                        candidate_read=candidate_read,
                        **kwargs,
                    )
                request_rows.append(request)
        requests = tuple(request_rows)
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
        len(exchanges) != len(requests)
        or tuple(row.request for row in exchanges) != requests
        or tuple(row.request_sha256 for row in exchanges)
        != tuple(request_sha256(row) for row in requests)
    ):
        raise QualificationRunnerError("T exchange coverage differs from the candidate cohort")
    exchange_index = 0
    for authority in value.candidates:
        quality_legs: list[
            tuple[
                ReferenceQualityVerdict,
                EvidenceArtifactRef,
                ReferenceQualityRawBinding,
                str,
            ]
        ] = []
        for candidate_read in range(1, quality_reads + 1):
            exchange = exchanges[exchange_index]
            exchange_index += 1
            if candidate_read == 1:
                raw = _raw_artifact(
                    lifecycle,
                    authority,
                    calibration,
                    selection,
                    reference_execution,
                    exchange,
                    judge_once,  # type: ignore[arg-type]
                )
            else:
                raw = _raw_artifact(
                    lifecycle,
                    authority,
                    calibration,
                    selection,
                    reference_execution,
                    exchange,
                    judge_once,  # type: ignore[arg-type]
                    candidate_read=candidate_read,
                )
            validation_kwargs = {
                "selected_delta_digest": authority.selected_delta_digest,
                "commitment": value.commitment,
                "entropy": entropy,
                "selection": selection,
                "calibration": calibration,
                "graph_requirement": _requirement(authority),
                "reference_execution": reference_execution,
                "reference_request_sha256": exchange.request_sha256,
            }
            if candidate_read != 1:
                validation_kwargs["candidate_read"] = candidate_read
            validate_quality_binding(
                authority.profile,
                raw,
                lifecycle,
                **validation_kwargs,
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
            quality_leg = score_reference_quality(
                quality_evidence,
                calibration=calibration,
                expected_context=value.calibration_context,
            )
            quality_legs.append(
                (quality_leg, raw_ref, raw.binding, exchange.request_sha256)
            )
        quality, raw_ref, raw_binding, t_request_sha256 = quality_legs[0]
        repeat_quality_verdict = quality_legs[1][0] if len(quality_legs) == 2 else None
        repeat_quality = (
            RepeatQualityWitness(
                quality_legs[1][0].evidence_digest,
                _decision(quality_legs[1][0].decision),
                quality_legs[1][0].candidate_mean_teacher_nll,
                quality_legs[1][1],
                quality_legs[1][2],
                quality_legs[1][3],
            )
            if len(quality_legs) == 2
            else None
        )
        if resident_speed_witness is not None:
            speed_witness: SpeedWitness | ResidentSpeedWitness = (
                resident_speed_witness
            )
            speed_grade, speedup = resident_speed_witness.regrade(
                calibration,
                value.calibration_context,
                expected_policy=value.speed_evidence_policy,
            )
            speed_evidence_digest = resident_speed_witness.evidence_digest
        else:
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
            speedup = format(speed.verdict.speedup, ".17g")
            speed_evidence_digest = speed.evidence_digest
        if speed_witness.policy != value.speed_evidence_policy:
            raise QualificationRunnerError("projected speed policy differs from authority")
        quality_grade = _quality_decision_pair(
            _decision(quality.decision),
            None if repeat_quality is None else repeat_quality.quality_decision,
        )
        candidate_mean_teacher_nll = max(
            (row[0].candidate_mean_teacher_nll for row in quality_legs),
            key=float,
        )
        candidate = next(
            row for row in lifecycle.candidates
            if row.arm.selected_delta_digest == authority.selected_delta_digest
        )
        audit_witness = audit_witnesses[authority.selected_delta_digest]
        if type(authority) is CandidateQualificationAuthority:
            graph = graph_grades[authority.selected_delta_digest]
            decision = _aggregate_decision(
                graph.decision, speed_grade, quality_grade,
                audit_witness.decision,
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
                speed_evidence_digest,
                speed_grade,
                speedup,
                quality.evidence_digest,
                quality_grade,
                candidate_mean_teacher_nll,
                raw_ref,
                raw_binding,
                speed_witness,
                t_request_sha256,
                audit_witness.digest,
                audit_witness.decision,
                audit_witness,
                decision,
                _report_reason(
                    graph, speed_grade, quality, audit_witness,
                    repeat_quality_verdict,
                ),
                decision is QualificationDecision.NO_DECISION,
                repeat_quality,
            ))
        else:
            if type(authority) is not DiscoveryCandidateQualificationAuthority:
                raise QualificationRunnerError(
                    "candidate authority has an unsupported type"
                )
            execution = discovery_grades[authority.selected_delta_digest]
            decision = _aggregate_decision(
                execution.decision, speed_grade, quality_grade,
                audit_witness.decision,
            )
            reports.append(DiscoveryCandidateQualificationReport(
                authority.selected_delta_digest,
                candidate.arm.digest,
                authority.execution_requirement.proposal_digest,
                candidate.candidate.launch.digest,
                authority.profile.digest,
                calibration.digest,
                execution,
                speed_evidence_digest,
                speed_grade,
                speedup,
                quality.evidence_digest,
                quality_grade,
                candidate_mean_teacher_nll,
                raw_ref,
                raw_binding,
                speed_witness,
                t_request_sha256,
                audit_witness.digest,
                audit_witness.decision,
                audit_witness,
                decision,
                _discovery_report_reason(
                    execution, speed_grade, quality, audit_witness,
                    repeat_quality_verdict,
                ),
                decision is QualificationDecision.NO_DECISION,
                repeat_quality,
            ))
    if exchange_index != len(exchanges):
        raise QualificationRunnerError("T exchange grouping differs from speed policy")
    attempt_type = (
        DiscoveryQualificationAttempt
        if all(
            type(row) is DiscoveryCandidateQualificationReport for row in reports
        )
        else CohortQualificationAttempt
    )
    attempt_args = (
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
    if resident_speed_witness is not None:
        if attempt_type is not CohortQualificationAttempt or len(reports) != 1:
            raise QualificationRunnerError(
                "resident qualification attempt changed lane or cardinality"
            )
        timing = QualificationTimingWitness(
            resident_speed_witness.resident_policy.digest,
            resident_speed_witness.evidence_digest,
            reports[0].audit_evidence_digest,
            reference_execution.session.digest,
            resident_speed_witness.resident_policy.max_qualification_seconds,
            resident_speed_witness.started_monotonic_s,
            resident_speed_witness.completed_monotonic_s,
            audit_started,
            audit_completed,
            t_pre.started_monotonic_s,
            t_post.completed_monotonic_s,
            teardown_after.observed_monotonic_s,
        )
        attempt = CohortQualificationAttempt(*attempt_args, timing)
    else:
        attempt = attempt_type(*attempt_args)
    reference = publish_causal_qualification(value.evidence_root, attempt)
    reopen_causal_qualification(value.evidence_root, reference, expected=value)
    return reference

__all__ = [
    "AuditWitness", "CandidateQualificationAuthority", "CandidateQualificationReport",
    "DiscoveryCandidateQualificationAuthority",
    "DiscoveryCandidateQualificationReport", "DiscoveryQualificationAttempt",
    "CausalQualificationInput", "CohortQualificationAttempt", "EntropyProvider",
    "HiddenJudge", "HiddenJudgeBinding", "HiddenJudgeReceipt",
    "QualificationRunnerError", "QualificationStageExit",
    "QualificationTimingWitness",
    "ReferenceExecutionWitness", "RepeatQualityWitness", "SpeedEvidencePolicy",
    "SpeedStageDisposition",
    "ResidentSpeedWitness", "SpeedWitness", "hidden_judge_output_digest",
    "publish_causal_qualification", "qualification_authority_digest",
    "publish_qualification_stage_exit", "reopen_causal_qualification",
    "reopen_qualification_stage_exit", "run_causal_qualification",
]
