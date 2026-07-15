"""Validator-owned arena admission, screening, and qualification planning.

An arena service is the trusted bridge between finalized publications and the
crownable qualification authority.  It binds the exact runtime/model/topology,
serving workload mixture, resource budgets, and reviewed provider identity.  Its
screens are deliberately non-economic: they may reject, retry, or promote a
candidate to qualification, but cannot produce a score or a crown.

The provider is an in-process validator object supplied by deployment code.  No
module path, entry point, or other miner-controlled dynamic import participates in
this authority boundary.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from optima.chain.publication import WorkerBundlePublication
from optima.eval.qualification_intake import (
    QualificationPlanFactory,
    QualificationReservation,
)
from optima.stack_identity import canonical_digest, require_sha256_hex


SERVICE_SCHEMA_VERSION = 1
WEIGHT_PPM = 1_000_000
SCREEN_STAGES = ("static", "build", "abi", "graph", "abbreviated_serving")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,255}\Z")
_ARCHITECTURE = re.compile(r"sm[0-9]{2,3}[a-z]?\Z")


class ArenaServiceError(ValueError):
    """Arena-service policy or provider output is inconsistent."""


def _digest(value: object, field: str) -> str:
    try:
        result = require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise ArenaServiceError(str(exc)) from None
    if result == "0" * 64:
        raise ArenaServiceError(f"{field} must not be the all-zero digest")
    return result


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ArenaServiceError(f"{field} is not a canonical identifier")
    return value


def _positive(value: object, field: str) -> int:
    if type(value) is not int or value < 1:
        raise ArenaServiceError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True)
class ArenaRuntimeIdentity:
    """Path-free serving identity shared by screening and qualification."""

    arena_id: str
    runtime_digest: str
    base_engine_digest: str
    validator_overlay_digest: str
    worker_distribution_digest: str
    model_revision_digest: str
    model_manifest_digest: str
    model_content_digest: str
    target_architecture: str
    topology_class: str
    topology_digest: str
    gpu_count: int
    tensor_parallel_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "arena_id", _identifier(self.arena_id, "arena_id"))
        for field in (
            "runtime_digest",
            "base_engine_digest",
            "validator_overlay_digest",
            "worker_distribution_digest",
            "model_revision_digest",
            "model_manifest_digest",
            "model_content_digest",
            "topology_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if not isinstance(self.target_architecture, str) or _ARCHITECTURE.fullmatch(
            self.target_architecture
        ) is None:
            raise ArenaServiceError("target_architecture must be canonical")
        object.__setattr__(
            self, "topology_class", _identifier(self.topology_class, "topology_class")
        )
        object.__setattr__(self, "gpu_count", _positive(self.gpu_count, "gpu_count"))
        object.__setattr__(
            self,
            "tensor_parallel_size",
            _positive(self.tensor_parallel_size, "tensor_parallel_size"),
        )
        if self.tensor_parallel_size > self.gpu_count:
            raise ArenaServiceError("tensor parallel size exceeds the GPU count")

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_id": self.arena_id,
            "base_engine_digest": self.base_engine_digest,
            "gpu_count": self.gpu_count,
            "model_content_digest": self.model_content_digest,
            "model_manifest_digest": self.model_manifest_digest,
            "model_revision_digest": self.model_revision_digest,
            "runtime_digest": self.runtime_digest,
            "target_architecture": self.target_architecture,
            "tensor_parallel_size": self.tensor_parallel_size,
            "topology_class": self.topology_class,
            "topology_digest": self.topology_digest,
            "validator_overlay_digest": self.validator_overlay_digest,
            "worker_distribution_digest": self.worker_distribution_digest,
        }


@dataclass(frozen=True)
class ServingShape:
    """One exact serving shape sampled within a workload regime."""

    input_tokens: int
    output_tokens: int
    batch_size: int
    samples: int

    def __post_init__(self) -> None:
        for field in ("input_tokens", "output_tokens", "batch_size", "samples"):
            object.__setattr__(self, field, _positive(getattr(self, field), field))

    def to_dict(self) -> dict[str, int]:
        return {
            "batch_size": self.batch_size,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "samples": self.samples,
        }


@dataclass(frozen=True)
class WorkloadRegime:
    name: str
    phase: str
    weight_ppm: int
    shapes: tuple[ServingShape, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, "workload regime name"))
        if self.phase not in {"decode", "long_prefill"}:
            raise ArenaServiceError("workload phase must be decode or long_prefill")
        if type(self.weight_ppm) is not int or not 1 <= self.weight_ppm <= WEIGHT_PPM:
            raise ArenaServiceError("workload regime weight_ppm is invalid")
        shapes = tuple(self.shapes)
        if (
            not shapes
            or any(type(row) is not ServingShape for row in shapes)
            or len({(row.input_tokens, row.output_tokens, row.batch_size) for row in shapes})
            != len(shapes)
        ):
            raise ArenaServiceError("workload regime shapes are empty or duplicated")
        object.__setattr__(self, "shapes", shapes)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "phase": self.phase,
            "shapes": [row.to_dict() for row in self.shapes],
            "weight_ppm": self.weight_ppm,
        }


@dataclass(frozen=True)
class WorkloadMixture:
    prompt_corpus_digest: str
    prompt_seed_scheme: str
    regimes: tuple[WorkloadRegime, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "prompt_corpus_digest",
            _digest(self.prompt_corpus_digest, "prompt_corpus_digest"),
        )
        object.__setattr__(
            self,
            "prompt_seed_scheme",
            _identifier(self.prompt_seed_scheme, "prompt_seed_scheme"),
        )
        regimes = tuple(self.regimes)
        if (
            not regimes
            or any(type(row) is not WorkloadRegime for row in regimes)
            or len({row.name for row in regimes}) != len(regimes)
            or sum(row.weight_ppm for row in regimes) != WEIGHT_PPM
            or {row.phase for row in regimes} != {"decode", "long_prefill"}
        ):
            raise ArenaServiceError(
                "workload mixture must uniquely cover decode and long_prefill at 1M ppm"
            )
        object.__setattr__(self, "regimes", regimes)

    @property
    def digest(self) -> str:
        return canonical_digest("optima.arena.workload-mixture", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "prompt_corpus_digest": self.prompt_corpus_digest,
            "prompt_seed_scheme": self.prompt_seed_scheme,
            "regimes": [row.to_dict() for row in self.regimes],
        }


@dataclass(frozen=True)
class ArenaCapacityPolicy:
    max_queue_depth: int
    max_queue_age_blocks: int
    max_active_screens: int
    max_active_qualifications: int
    max_cohort_size: int
    screen_retry_limit: int
    qualification_retry_limit: int
    infrastructure_retry_limit: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            object.__setattr__(self, field, _positive(getattr(self, field), field))
        if self.max_cohort_size > self.max_active_qualifications:
            raise ArenaServiceError("qualification cohort exceeds active capacity")

    def to_dict(self) -> dict[str, int]:
        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass(frozen=True)
class ScreenStagePolicy:
    stage: str
    timeout_ms: int

    def __post_init__(self) -> None:
        if self.stage not in SCREEN_STAGES:
            raise ArenaServiceError("screen stage is unsupported")
        object.__setattr__(self, "timeout_ms", _positive(self.timeout_ms, "timeout_ms"))

    def to_dict(self) -> dict[str, object]:
        return {"stage": self.stage, "timeout_ms": self.timeout_ms}


@dataclass(frozen=True)
class NonCrownScreenPolicy:
    stages: tuple[ScreenStagePolicy, ...]

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if any(type(row) is not ScreenStagePolicy for row in stages) or tuple(
            row.stage for row in stages
        ) != SCREEN_STAGES:
            raise ArenaServiceError("non-crown screen stages or order differ")
        object.__setattr__(self, "stages", stages)

    def to_dict(self) -> dict[str, object]:
        return {
            "crownable": False,
            "stages": [row.to_dict() for row in self.stages],
        }


@dataclass(frozen=True)
class ArenaServiceManifest:
    runtime: ArenaRuntimeIdentity
    workload: WorkloadMixture
    capacity: ArenaCapacityPolicy
    screens: NonCrownScreenPolicy
    qualification_policy_digest: str
    provider_digest: str
    schema_version: int = SERVICE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.runtime) is not ArenaRuntimeIdentity
            or type(self.workload) is not WorkloadMixture
            or type(self.capacity) is not ArenaCapacityPolicy
            or type(self.screens) is not NonCrownScreenPolicy
            or type(self.schema_version) is not int
            or self.schema_version != SERVICE_SCHEMA_VERSION
        ):
            raise ArenaServiceError("arena service manifest is not exactly typed")
        object.__setattr__(
            self,
            "qualification_policy_digest",
            _digest(self.qualification_policy_digest, "qualification_policy_digest"),
        )
        object.__setattr__(
            self, "provider_digest", _digest(self.provider_digest, "provider_digest")
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.arena.service", self.to_dict())

    @property
    def service_id(self) -> str:
        return f"{self.runtime.arena_id}@{self.digest}"

    def to_dict(self) -> dict[str, object]:
        return {
            "capacity": self.capacity.to_dict(),
            "provider_digest": self.provider_digest,
            "qualification_policy_digest": self.qualification_policy_digest,
            "runtime": self.runtime.to_dict(),
            "schema_version": self.schema_version,
            "screens": self.screens.to_dict(),
            "workload": self.workload.to_dict(),
        }


@dataclass(frozen=True)
class ArenaQueueSnapshot:
    queued: int
    oldest_age_blocks: int
    active_screens: int
    active_qualifications: int

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise ArenaServiceError(f"queue snapshot {field} must be nonnegative")


class AdmissionDecision(str, Enum):
    ADMIT = "admit"
    QUEUE = "queue"
    HOLD = "hold"


class ScreenGrade(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NO_DECISION = "no_decision"


class PromotionDecision(str, Enum):
    PROMOTE = "promote"
    REJECT = "reject"
    RETRY = "retry"
    HOLD = "hold"


@dataclass(frozen=True)
class ArenaCandidateBinding:
    """Trusted local binding whose digest excludes the validator host path."""

    reservation: QualificationReservation
    publication: WorkerBundlePublication
    screen_attempt: int

    def __post_init__(self) -> None:
        if (
            type(self.reservation) is not QualificationReservation
            or type(self.publication) is not WorkerBundlePublication
        ):
            raise ArenaServiceError("candidate binding is not exactly typed")
        object.__setattr__(
            self, "screen_attempt", _positive(self.screen_attempt, "screen_attempt")
        )
        if self.reservation.submission_digest != self.publication.digest:
            raise ArenaServiceError("candidate publication differs from reservation")

    @property
    def digest(self) -> str:
        reservation = self.reservation.to_dict()
        # Cohort position is assigned only after independent non-crown screens
        # finish.  It cannot make the same finalized candidate acquire a new
        # screen identity when a neighbor is rejected before qualification.
        reservation.pop("arrival_order")
        return canonical_digest(
            "optima.arena.candidate-binding",
            {
                "publication_digest": self.publication.digest,
                "reservation": reservation,
                "screen_attempt": self.screen_attempt,
            },
        )


@dataclass(frozen=True)
class ScreenStageResult:
    stage: str
    grade: ScreenGrade
    evidence_digest: str
    elapsed_ms: int

    def __post_init__(self) -> None:
        if self.stage not in SCREEN_STAGES or type(self.grade) is not ScreenGrade:
            raise ArenaServiceError("screen result stage or grade is invalid")
        object.__setattr__(
            self, "evidence_digest", _digest(self.evidence_digest, "screen evidence")
        )
        object.__setattr__(self, "elapsed_ms", _positive(self.elapsed_ms, "elapsed_ms"))

    def to_dict(self) -> dict[str, object]:
        return {
            "elapsed_ms": self.elapsed_ms,
            "evidence_digest": self.evidence_digest,
            "grade": self.grade.value,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class ArenaScreenReceipt:
    service_digest: str
    candidate_digest: str
    screen_attempt: int
    results: tuple[ScreenStageResult, ...]
    decision: PromotionDecision

    def __post_init__(self) -> None:
        for field in ("service_digest", "candidate_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(
            self, "screen_attempt", _positive(self.screen_attempt, "screen_attempt")
        )
        results = tuple(self.results)
        grades = tuple(row.grade for row in results)
        if (
            not results
            or any(type(row) is not ScreenStageResult for row in results)
            or tuple(row.stage for row in results) != SCREEN_STAGES[: len(results)]
            or type(self.decision) is not PromotionDecision
            or any(grade is not ScreenGrade.PASS for grade in grades[:-1])
        ):
            raise ArenaServiceError("screen receipt results are not a canonical prefix")
        terminal = grades[-1]
        if (
            (self.decision is PromotionDecision.PROMOTE
             and (len(results) != len(SCREEN_STAGES) or terminal is not ScreenGrade.PASS))
            or (self.decision is PromotionDecision.REJECT and terminal is not ScreenGrade.FAIL)
            or (
                self.decision in {PromotionDecision.RETRY, PromotionDecision.HOLD}
                and terminal is not ScreenGrade.NO_DECISION
            )
        ):
            raise ArenaServiceError("screen decision is not derived from stage evidence")
        object.__setattr__(self, "results", results)

    @property
    def digest(self) -> str:
        return canonical_digest("optima.arena.screen-receipt", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_digest": self.candidate_digest,
            "decision": self.decision.value,
            "results": [row.to_dict() for row in self.results],
            "screen_attempt": self.screen_attempt,
            "service_digest": self.service_digest,
        }


@dataclass(frozen=True)
class ArenaQualificationRequest:
    service_digest: str
    qualification_policy_digest: str
    candidates: tuple[ArenaCandidateBinding, ...]
    screen_receipts: tuple[ArenaScreenReceipt, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "service_digest", _digest(self.service_digest, "service_digest")
        )
        object.__setattr__(
            self,
            "qualification_policy_digest",
            _digest(self.qualification_policy_digest, "qualification_policy_digest"),
        )
        candidates = tuple(self.candidates)
        receipts = tuple(self.screen_receipts)
        if (
            not candidates
            or any(type(row) is not ArenaCandidateBinding for row in candidates)
            or any(type(row) is not ArenaScreenReceipt for row in receipts)
            or len(candidates) != len(receipts)
            or tuple(row.candidate_digest for row in receipts)
            != tuple(row.digest for row in candidates)
            or any(
                row.service_digest != self.service_digest
                or row.decision is not PromotionDecision.PROMOTE
                for row in receipts
            )
        ):
            raise ArenaServiceError("qualification request lacks exact promoted coverage")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "screen_receipts", receipts)


@dataclass(frozen=True)
class ArenaQualificationWork:
    factory: QualificationPlanFactory
    executor: object
    entropy_provider: object
    hidden_judge: object
    deadline: float
    qualification_policy_digest: str

    def __post_init__(self) -> None:
        if type(self.factory) is not QualificationPlanFactory:
            raise ArenaServiceError("qualification work has no exact plan factory")
        object.__setattr__(
            self,
            "qualification_policy_digest",
            _digest(self.qualification_policy_digest, "qualification_policy_digest"),
        )
        if not callable(self.entropy_provider) or not callable(self.hidden_judge):
            raise ArenaServiceError("qualification work authorities are not callable")
        if (
            isinstance(self.deadline, bool)
            or not isinstance(self.deadline, (int, float))
            or not math.isfinite(float(self.deadline))
            or float(self.deadline) <= 0
        ):
            raise ArenaServiceError("qualification deadline is invalid")


class ArenaServiceProvider(Protocol):
    """Reviewed deployment adapter; never resolved from submission metadata."""

    provider_digest: str

    def run_screen(
        self,
        manifest: ArenaServiceManifest,
        stage: ScreenStagePolicy,
        candidate: ArenaCandidateBinding,
    ) -> ScreenStageResult: ...

    def build_qualification(
        self, request: ArenaQualificationRequest, state: object | None = None
    ) -> ArenaQualificationWork: ...


class ArenaService:
    """Pure validator authority used by the durable intake controller."""

    def __init__(self, manifest: ArenaServiceManifest, provider: ArenaServiceProvider):
        if type(manifest) is not ArenaServiceManifest:
            raise ArenaServiceError("service manifest is not exactly typed")
        if _digest(getattr(provider, "provider_digest", None), "provider_digest") != (
            manifest.provider_digest
        ):
            raise ArenaServiceError("provider implementation identity differs")
        if not callable(getattr(provider, "run_screen", None)) or not callable(
            getattr(provider, "build_qualification", None)
        ):
            raise ArenaServiceError("provider does not implement the trusted interface")
        self.manifest = manifest
        self._provider = provider

    @property
    def identity(self) -> str:
        return self.manifest.digest

    def admit(self, state: ArenaQueueSnapshot) -> AdmissionDecision:
        if type(state) is not ArenaQueueSnapshot:
            raise ArenaServiceError("queue state is not exactly typed")
        policy = self.manifest.capacity
        if (
            state.queued >= policy.max_queue_depth
            or state.oldest_age_blocks >= policy.max_queue_age_blocks
        ):
            return AdmissionDecision.HOLD
        if state.active_screens >= policy.max_active_screens:
            return AdmissionDecision.QUEUE
        return AdmissionDecision.ADMIT

    def admit_qualification(
        self, state: ArenaQueueSnapshot, *, cohort_size: int
    ) -> AdmissionDecision:
        if type(state) is not ArenaQueueSnapshot:
            raise ArenaServiceError("queue state is not exactly typed")
        size = _positive(cohort_size, "cohort_size")
        policy = self.manifest.capacity
        if size > policy.max_cohort_size:
            return AdmissionDecision.HOLD
        if (
            state.queued >= policy.max_queue_depth
            or state.oldest_age_blocks >= policy.max_queue_age_blocks
        ):
            return AdmissionDecision.HOLD
        if state.active_qualifications + size > policy.max_active_qualifications:
            return AdmissionDecision.QUEUE
        return AdmissionDecision.ADMIT

    def retry_disposition(
        self, lane: str, *, attempt: int
    ) -> PromotionDecision:
        current = _positive(attempt, "attempt")
        budgets = {
            "screen": self.manifest.capacity.screen_retry_limit,
            "qualification": self.manifest.capacity.qualification_retry_limit,
            "infrastructure": self.manifest.capacity.infrastructure_retry_limit,
        }
        if lane not in budgets:
            raise ArenaServiceError("retry lane is unsupported")
        return (
            PromotionDecision.RETRY
            if current < budgets[lane]
            else PromotionDecision.HOLD
        )

    def screen(self, candidate: ArenaCandidateBinding) -> ArenaScreenReceipt:
        if type(candidate) is not ArenaCandidateBinding:
            raise ArenaServiceError("screen candidate is not exactly typed")
        results: list[ScreenStageResult] = []
        decision = PromotionDecision.PROMOTE
        for stage in self.manifest.screens.stages:
            result = self._provider.run_screen(self.manifest, stage, candidate)
            if type(result) is not ScreenStageResult or result.stage != stage.stage:
                raise ArenaServiceError("provider changed the requested screen stage")
            if result.elapsed_ms > stage.timeout_ms:
                result = ScreenStageResult(
                    result.stage,
                    ScreenGrade.NO_DECISION,
                    result.evidence_digest,
                    result.elapsed_ms,
                )
            results.append(result)
            if result.grade is ScreenGrade.FAIL:
                decision = PromotionDecision.REJECT
                break
            if result.grade is ScreenGrade.NO_DECISION:
                decision = self.retry_disposition(
                    "screen", attempt=candidate.screen_attempt
                )
                break
        return ArenaScreenReceipt(
            self.identity,
            candidate.digest,
            candidate.screen_attempt,
            tuple(results),
            decision,
        )

    def plan_qualification(
        self,
        candidates: tuple[ArenaCandidateBinding, ...],
        screen_receipts: tuple[ArenaScreenReceipt, ...],
        *,
        state: object | None = None,
    ) -> ArenaQualificationWork:
        if len(candidates) > self.manifest.capacity.max_cohort_size:
            raise ArenaServiceError("qualification cohort exceeds arena capacity")
        request = ArenaQualificationRequest(
            self.identity,
            self.manifest.qualification_policy_digest,
            tuple(candidates),
            tuple(screen_receipts),
        )
        work = self._provider.build_qualification(request, state)
        if type(work) is not ArenaQualificationWork:
            raise ArenaServiceError("provider returned untyped qualification work")
        if work.qualification_policy_digest != request.qualification_policy_digest:
            raise ArenaServiceError("provider changed the qualification policy")
        expected = tuple(row.reservation for row in request.candidates)
        if work.factory.manifest.reservations != expected:
            raise ArenaServiceError("provider changed finalized qualification order")
        return work


class ArenaServiceRegistry:
    """Closed validator configuration for registered arena services."""

    def __init__(self, services: tuple[ArenaService, ...]):
        rows = tuple(services)
        if not rows or any(type(row) is not ArenaService for row in rows):
            raise ArenaServiceError("arena service registry is empty or ambiguous")
        arena_ids = tuple(row.manifest.runtime.arena_id for row in rows)
        if (
            arena_ids != tuple(sorted(arena_ids))
            or len(set(arena_ids)) != len(arena_ids)
            or len({row.identity for row in rows}) != len(rows)
        ):
            raise ArenaServiceError("arena service registry is empty or ambiguous")
        self._services = rows

    @property
    def digest(self) -> str:
        return canonical_digest(
            "optima.arena.service-registry",
            {
                "services": [
                    {
                        "arena_id": row.manifest.runtime.arena_id,
                        "service_digest": row.identity,
                    }
                    for row in self._services
                ]
            },
        )

    def require(self, arena_id: str) -> ArenaService:
        expected = _identifier(arena_id, "arena_id")
        for service in self._services:
            if service.manifest.runtime.arena_id == expected:
                return service
        raise ArenaServiceError(f"arena {expected!r} is not registered")


__all__ = [
    "AdmissionDecision",
    "ArenaCandidateBinding",
    "ArenaCapacityPolicy",
    "ArenaQualificationRequest",
    "ArenaQualificationWork",
    "ArenaQueueSnapshot",
    "ArenaRuntimeIdentity",
    "ArenaScreenReceipt",
    "ArenaService",
    "ArenaServiceError",
    "ArenaServiceManifest",
    "ArenaServiceProvider",
    "ArenaServiceRegistry",
    "NonCrownScreenPolicy",
    "PromotionDecision",
    "SCREEN_STAGES",
    "ScreenGrade",
    "ScreenStagePolicy",
    "ScreenStageResult",
    "ServingShape",
    "WorkloadMixture",
    "WorkloadRegime",
]
