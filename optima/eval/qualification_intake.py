"""Finalized-intake projection for causal qualification evidence.

This module is deliberately narrower than the qualification runner.  It binds an
already prepared validator-owned plan to finalized reservation identities, turns
exact per-shape graph observations into the runner's canonical raw evidence, and
projects a completed attempt (or a retryable cohort failure) into per-reservation
three-way outcomes.  It does not fetch submissions, execute candidate code, settle
scores, or publish weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from optima.settlement import SettlementCandidate

from optima.discovery import DiscoveryArmPlan
from optima.eval.evidence_store import EvidenceArtifactRef, publish_evidence
from optima.eval.qualification import (
    GRAPH_EVIDENCE_DOMAIN,
    GRAPH_EVIDENCE_MEDIA_TYPE,
    GRAPH_EVIDENCE_SCHEMA,
    GraphMemberEvidence,
    GraphShapeEvidence,
    GraphVariantEvidence,
    GraphVerificationEvidenceRef,
    GraphVerificationGrade,
    GraphVerificationRawEvidence,
    GraphVerificationRequirement,
    QualificationDecision,
    regrade_graph_verification,
    reopen_graph_verification,
)
from optima.eval.qualification_runner import (
    CandidateQualificationReport,
    CausalQualificationInput,
    CohortQualificationAttempt,
    DiscoveryCandidateQualificationReport,
    DiscoveryQualificationAttempt,
    QualificationRunnerError,
    qualification_authority_digest,
    reopen_causal_qualification,
    run_causal_qualification,
)
from optima.eval.scoring import RawSpeedEvidenceError
from optima.stack_identity import (
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


AUTHORITY_SCHEMA_VERSION = 1
_LANES = frozenset({"registered", "discovery"})
_RETRY_STRATEGIES = frozenset({"requeue", "bisect"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")


class QualificationIntakeError(ValueError):
    """Finalized qualification authority or evidence is inconsistent."""


def _digest(value: object, field: str) -> str:
    try:
        result = require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise QualificationIntakeError(str(exc)) from None
    if result == "0" * 64:
        raise QualificationIntakeError(f"{field} must not be the all-zero digest")
    return result


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise QualificationIntakeError(f"{field} is not a canonical identifier")
    return value


def _integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise QualificationIntakeError(f"{field} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class QualificationReservation:
    """One finalized submission in immutable cohort order."""

    reservation_digest: str
    submission_digest: str
    target_id: str
    selected_delta_digest: str
    arrival_order: int
    hotkey: str
    finalized_block: int
    finalized_event_index: int
    finalized_event_subindex: int
    target_members: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "reservation_digest",
            "submission_digest",
            "selected_delta_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "target_id", _identifier(self.target_id, "target_id"))
        object.__setattr__(
            self, "arrival_order", _integer(self.arrival_order, "arrival_order")
        )
        if (
            not isinstance(self.hotkey, str)
            or not self.hotkey
            or self.hotkey.strip() != self.hotkey
            or len(self.hotkey) > 256
            or any(char in self.hotkey for char in "\x00\r\n")
        ):
            raise QualificationIntakeError("reservation hotkey is malformed")
        for field in (
            "finalized_block",
            "finalized_event_index",
            "finalized_event_subindex",
        ):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        members = tuple(self.target_members)
        if (
            not members
            or members != tuple(sorted(set(members)))
            or any(_identifier(member, "target member") != member for member in members)
        ):
            raise QualificationIntakeError("reservation target members are not canonical")
        object.__setattr__(self, "target_members", members)

    def to_dict(self) -> dict[str, object]:
        return {
            "arrival_order": self.arrival_order,
            "finalized_block": self.finalized_block,
            "finalized_event_index": self.finalized_event_index,
            "finalized_event_subindex": self.finalized_event_subindex,
            "hotkey": self.hotkey,
            "reservation_digest": self.reservation_digest,
            "selected_delta_digest": self.selected_delta_digest,
            "submission_digest": self.submission_digest,
            "target_id": self.target_id,
            "target_members": list(self.target_members),
        }

    @classmethod
    def from_dict(cls, value: object) -> "QualificationReservation":
        fields = {
            "arrival_order",
            "finalized_block",
            "finalized_event_index",
            "finalized_event_subindex",
            "hotkey",
            "reservation_digest",
            "selected_delta_digest",
            "submission_digest",
            "target_id",
            "target_members",
        }
        if type(value) is not dict or set(value) != fields:
            raise QualificationIntakeError("reservation fields do not match the schema")
        if type(value["target_members"]) is not list:
            raise QualificationIntakeError("reservation target members are malformed")
        return cls(**{**value, "target_members": tuple(value["target_members"])})  # type: ignore[arg-type]


@dataclass(frozen=True)
class QualificationAuthorityManifest:
    """Public identity for one private, validator-owned qualification plan.

    ``selection_secret_reference`` names a record in a private secret store.  The
    secret bytes themselves never enter this serializable object.
    """

    lane: str
    authority_digest: str
    source_digest: str
    commitment_digest: str
    selection_secret_reference: str
    candidate_deltas: tuple[str, ...]
    reservations: tuple[QualificationReservation, ...]
    schema_version: int = AUTHORITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.lane not in _LANES:
            raise QualificationIntakeError("qualification lane is unsupported")
        for field in (
            "authority_digest",
            "source_digest",
            "commitment_digest",
            "selection_secret_reference",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        deltas = tuple(_digest(row, "candidate delta") for row in self.candidate_deltas)
        reservations = tuple(self.reservations)
        if (
            type(self.schema_version) is not int
            or self.schema_version != AUTHORITY_SCHEMA_VERSION
            or not deltas
            or len(set(deltas)) != len(deltas)
            or any(type(row) is not QualificationReservation for row in reservations)
            or len(reservations) != len(deltas)
            or tuple(row.selected_delta_digest for row in reservations) != deltas
            or len({row.reservation_digest for row in reservations}) != len(reservations)
            or len({row.arrival_order for row in reservations}) != len(reservations)
            or (self.lane == "discovery" and len(reservations) != 1)
        ):
            raise QualificationIntakeError(
                "qualification reservations do not exactly bind the candidate order"
            )
        object.__setattr__(self, "candidate_deltas", deltas)
        object.__setattr__(self, "reservations", reservations)

    @classmethod
    def seal(
        cls,
        value: CausalQualificationInput,
        *,
        reservations: tuple[QualificationReservation, ...],
        selection_secret_reference: str,
    ) -> "QualificationAuthorityManifest":
        if type(value) is not CausalQualificationInput:
            raise QualificationIntakeError("qualification plan is not exactly typed")
        discovery = type(value.prepared.source) is DiscoveryArmPlan
        return cls(
            "discovery" if discovery else "registered",
            qualification_authority_digest(value),
            value.prepared.source.digest,
            value.commitment.digest,
            selection_secret_reference,
            tuple(row.selected_delta_digest for row in value.candidates),
            reservations,
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.qualification.intake-authority", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "authority_digest": self.authority_digest,
            "candidate_deltas": list(self.candidate_deltas),
            "commitment_digest": self.commitment_digest,
            "lane": self.lane,
            "reservations": [row.to_dict() for row in self.reservations],
            "schema_version": self.schema_version,
            "selection_secret_reference": self.selection_secret_reference,
            "source_digest": self.source_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "QualificationAuthorityManifest":
        fields = {
            "authority_digest",
            "candidate_deltas",
            "commitment_digest",
            "lane",
            "reservations",
            "schema_version",
            "selection_secret_reference",
            "source_digest",
        }
        if type(value) is not dict or set(value) != fields:
            raise QualificationIntakeError("authority manifest fields do not match")
        if type(value["candidate_deltas"]) is not list or type(value["reservations"]) is not list:
            raise QualificationIntakeError("authority manifest arrays are malformed")
        return cls(
            lane=value["lane"],  # type: ignore[arg-type]
            authority_digest=value["authority_digest"],  # type: ignore[arg-type]
            source_digest=value["source_digest"],  # type: ignore[arg-type]
            commitment_digest=value["commitment_digest"],  # type: ignore[arg-type]
            selection_secret_reference=value["selection_secret_reference"],  # type: ignore[arg-type]
            candidate_deltas=tuple(value["candidate_deltas"]),  # type: ignore[arg-type]
            reservations=tuple(
                QualificationReservation.from_dict(row)
                for row in value["reservations"]  # type: ignore[union-attr]
            ),
            schema_version=value["schema_version"],  # type: ignore[arg-type]
        )


SecretLoader = Callable[[str], bytes]
PlanBuilder = Callable[[bytes], CausalQualificationInput]


@dataclass(frozen=True)
class QualificationPlanFactory:
    """Resolve a private secret and reconstruct one exact public authority manifest."""

    manifest: QualificationAuthorityManifest
    secret_loader: SecretLoader
    plan_builder: PlanBuilder

    def __post_init__(self) -> None:
        if type(self.manifest) is not QualificationAuthorityManifest:
            raise QualificationIntakeError("plan factory manifest is not exactly typed")
        if not callable(self.secret_loader) or not callable(self.plan_builder):
            raise QualificationIntakeError("plan factory authorities must be callable")

    def build(self) -> CausalQualificationInput:
        secret = self.secret_loader(self.manifest.selection_secret_reference)
        if type(secret) is not bytes or len(secret) < 32:
            raise QualificationIntakeError("private selection secret is unavailable")
        value = self.plan_builder(secret)
        if type(value) is not CausalQualificationInput:
            raise QualificationIntakeError("plan builder returned an untyped plan")
        if value.selection_secret != secret:
            raise QualificationIntakeError("plan builder substituted the private secret")
        observed = QualificationAuthorityManifest.seal(
            value,
            reservations=self.manifest.reservations,
            selection_secret_reference=self.manifest.selection_secret_reference,
        )
        if observed != self.manifest:
            raise QualificationIntakeError("rebuilt qualification authority differs")
        return value


@dataclass(frozen=True)
class GraphShapeObservation:
    """Non-aggregate facts for one validator-named verification shape."""

    descriptor_digest: str
    applicable: bool
    eager_passed: bool
    capture_succeeded: bool
    replay_count: int
    replay_passed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "descriptor_digest", _digest(self.descriptor_digest, "shape descriptor")
        )
        for field in (
            "applicable",
            "eager_passed",
            "capture_succeeded",
            "replay_passed",
        ):
            if type(getattr(self, field)) is not bool:
                raise QualificationIntakeError(f"{field} must be an exact boolean")
        object.__setattr__(self, "replay_count", _integer(self.replay_count, "replay_count"))
        if (
            (not self.applicable and any(
                (self.eager_passed, self.capture_succeeded, self.replay_passed, self.replay_count)
            ))
            or (not self.eager_passed and any(
                (self.capture_succeeded, self.replay_passed, self.replay_count)
            ))
            or (not self.capture_succeeded and any((self.replay_passed, self.replay_count)))
            or (self.replay_passed and self.replay_count < 1)
        ):
            raise QualificationIntakeError("graph shape observation is causally inconsistent")

    @property
    def failure_kind(self) -> str:
        if not self.applicable:
            return "not_applicable"
        if not self.eager_passed:
            return "eager"
        if not self.capture_succeeded:
            return "capture"
        if not self.replay_passed:
            return "replay"
        return "none"

    def evidence(self) -> GraphShapeEvidence:
        return GraphShapeEvidence(
            self.descriptor_digest,
            self.applicable,
            self.eager_passed,
            self.applicable,
            self.replay_count,
            self.replay_passed,
            self.failure_kind,
        )


@dataclass(frozen=True)
class GraphVariantObservation:
    slot_id: str
    variant_id: str
    context_applicable: bool
    domain_coverage_complete: bool
    shapes: tuple[GraphShapeObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id", _identifier(self.slot_id, "slot_id"))
        object.__setattr__(self, "variant_id", _identifier(self.variant_id, "variant_id"))
        for field in ("context_applicable", "domain_coverage_complete"):
            if type(getattr(self, field)) is not bool:
                raise QualificationIntakeError(f"{field} must be an exact boolean")
        shapes = tuple(self.shapes)
        if not shapes or any(type(row) is not GraphShapeObservation for row in shapes):
            raise QualificationIntakeError("graph variant requires exact per-shape facts")
        object.__setattr__(self, "shapes", shapes)


@dataclass(frozen=True)
class GraphMemberObservation:
    slot_id: str
    variants: tuple[GraphVariantObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id", _identifier(self.slot_id, "slot_id"))
        variants = tuple(self.variants)
        if not variants or any(
            type(row) is not GraphVariantObservation or row.slot_id != self.slot_id
            for row in variants
        ):
            raise QualificationIntakeError("graph member variants are incomplete")
        object.__setattr__(self, "variants", variants)


@dataclass(frozen=True)
class GraphVerificationObservation:
    requirement_digest: str
    members: tuple[GraphMemberObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "requirement_digest", _digest(self.requirement_digest, "requirement")
        )
        members = tuple(self.members)
        if not members or any(type(row) is not GraphMemberObservation for row in members):
            raise QualificationIntakeError(
                "graph observation must contain exact member/variant/shape facts"
            )
        object.__setattr__(self, "members", members)


@dataclass(frozen=True)
class GraphEvidenceProduct:
    requirement_digest: str
    artifact_ref: EvidenceArtifactRef
    evidence_ref: GraphVerificationEvidenceRef
    raw_evidence_digest: str
    grade: GraphVerificationGrade

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "requirement_digest", _digest(self.requirement_digest, "requirement")
        )
        object.__setattr__(
            self, "raw_evidence_digest", _digest(self.raw_evidence_digest, "raw evidence")
        )
        if (
            type(self.artifact_ref) is not EvidenceArtifactRef
            or type(self.evidence_ref) is not GraphVerificationEvidenceRef
            or type(self.grade) is not GraphVerificationGrade
            or self.artifact_ref.domain != GRAPH_EVIDENCE_DOMAIN
            or self.artifact_ref.media_type != GRAPH_EVIDENCE_MEDIA_TYPE
            or self.artifact_ref.schema != GRAPH_EVIDENCE_SCHEMA
            or self.evidence_ref.requirement_digest != self.requirement_digest
            or self.evidence_ref.raw_evidence_digest != self.raw_evidence_digest
            or self.grade.requirement_digest != self.requirement_digest
            or self.grade.evidence_ref_digest != self.evidence_ref.digest
            or self.grade.raw_evidence_digest != self.raw_evidence_digest
        ):
            raise QualificationIntakeError("published graph evidence identities differ")


def publish_graph_observation(
    evidence_root,
    requirement: GraphVerificationRequirement,
    observation: GraphVerificationObservation,
) -> GraphEvidenceProduct:
    """Publish raw graph facts without accepting a verifier aggregate verdict."""

    if type(requirement) is not GraphVerificationRequirement:
        raise QualificationIntakeError("graph requirement is not exactly typed")
    if type(observation) is not GraphVerificationObservation:
        raise QualificationIntakeError(
            "graph evidence requires exact observations, not VerifyResult or booleans"
        )
    if observation.requirement_digest != requirement.digest:
        raise QualificationIntakeError("graph observation names another requirement")
    expected_members = tuple(row.slot_id for row in requirement.binding.members)
    if tuple(row.slot_id for row in observation.members) != expected_members:
        raise QualificationIntakeError("graph member observations differ from the requirement")
    required_by_member: dict[str, tuple] = {}
    for row in requirement.variants:
        required_by_member[row.slot_id] = required_by_member.get(row.slot_id, ()) + (row,)
    members = []
    for observed_member in observation.members:
        required_variants = required_by_member[observed_member.slot_id]
        if tuple(row.variant_id for row in observed_member.variants) != tuple(
            row.variant_id for row in required_variants
        ):
            raise QualificationIntakeError("graph variant observations differ")
        variants = []
        for observed, required in zip(
            observed_member.variants, required_variants, strict=True
        ):
            if tuple(row.descriptor_digest for row in observed.shapes) != (
                required.shape_descriptor_digests
            ):
                raise QualificationIntakeError("graph shape observations differ")
            variants.append(
                GraphVariantEvidence(
                    observed.slot_id,
                    observed.variant_id,
                    observed.context_applicable,
                    observed.domain_coverage_complete,
                    tuple(row.evidence() for row in observed.shapes),
                )
            )
        members.append(GraphMemberEvidence(observed_member.slot_id, tuple(variants)))
    raw = GraphVerificationRawEvidence(requirement.digest, tuple(members))
    evidence_ref = GraphVerificationEvidenceRef(
        requirement.binding, requirement.digest, raw.digest
    )
    artifact_ref = publish_evidence(
        evidence_root,
        canonical_json_bytes(raw.to_dict()),
        domain=GRAPH_EVIDENCE_DOMAIN,
        media_type=GRAPH_EVIDENCE_MEDIA_TYPE,
        schema=GRAPH_EVIDENCE_SCHEMA,
    )
    grade = reopen_graph_verification(
        evidence_root, artifact_ref, requirement, evidence_ref
    )
    if grade != regrade_graph_verification(requirement, evidence_ref, raw):
        raise QualificationIntakeError("published graph evidence regraded differently")
    return GraphEvidenceProduct(
        requirement.digest, artifact_ref, evidence_ref, raw.digest, grade
    )


@dataclass(frozen=True)
class QualificationRetryPlan:
    authority_manifest_digest: str
    strategy: str
    reservation_groups: tuple[tuple[str, ...], ...]
    failure_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_manifest_digest",
            _digest(self.authority_manifest_digest, "authority manifest"),
        )
        object.__setattr__(self, "failure_digest", _digest(self.failure_digest, "failure"))
        if self.strategy not in _RETRY_STRATEGIES:
            raise QualificationIntakeError("retry strategy is unsupported")
        groups = tuple(tuple(group) for group in self.reservation_groups)
        flat = tuple(row for group in groups for row in group)
        if (
            not groups
            or any(not group for group in groups)
            or any(_digest(row, "retry reservation") != row for row in flat)
            or len(set(flat)) != len(flat)
            or (self.strategy == "bisect" and len(groups) != 2)
        ):
            raise QualificationIntakeError("retry groups are malformed")
        object.__setattr__(self, "reservation_groups", groups)


@dataclass(frozen=True)
class QualificationIntakeOutcome:
    reservation_digest: str
    selected_delta_digest: str
    authority_manifest_digest: str
    decision: QualificationDecision
    reason: str
    retryable: bool
    attempt_artifact_sha256: str | None = None
    report_digest: str | None = None
    failure_digest: str | None = None
    settlement_candidate: SettlementCandidate | None = None

    def __post_init__(self) -> None:
        for field in (
            "reservation_digest",
            "selected_delta_digest",
            "authority_manifest_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if type(self.decision) is not QualificationDecision:
            raise QualificationIntakeError("outcome decision is not typed")
        object.__setattr__(self, "reason", _identifier(self.reason, "reason"))
        if type(self.retryable) is not bool or self.retryable != (
            self.decision is QualificationDecision.NO_DECISION
        ):
            raise QualificationIntakeError("outcome retryability disagrees with decision")
        for field in ("attempt_artifact_sha256", "report_digest", "failure_digest"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _digest(value, field))
        if (self.report_digest is None) != (self.attempt_artifact_sha256 is None):
            raise QualificationIntakeError("outcome report and attempt coverage differ")
        if self.failure_digest is not None and self.report_digest is not None:
            raise QualificationIntakeError("outcome cannot be both report and failure based")
        report_based = self.report_digest is not None
        failure_based = self.failure_digest is not None
        if self.decision is QualificationDecision.NO_DECISION:
            if report_based == failure_based:
                raise QualificationIntakeError(
                    "NO_DECISION must retain exactly one report or failure product"
                )
        elif not report_based or failure_based:
            raise QualificationIntakeError(
                "PASS/FAIL requires a complete attempt and report product"
            )
        from optima.settlement import SettlementCandidate

        if self.settlement_candidate is not None:
            if type(self.settlement_candidate) is not SettlementCandidate:
                raise QualificationIntakeError(
                    "settlement projection is not exactly typed"
                )
            if self.decision is not QualificationDecision.PASS:
                raise QualificationIntakeError(
                    "non-PASS outcome cannot carry settlement authority"
                )
            if (
                self.settlement_candidate.reservation_digest != self.reservation_digest
                or self.settlement_candidate.selected_delta_digest
                != self.selected_delta_digest
                or self.settlement_candidate.qualification_authority_digest
                != self.authority_manifest_digest
                or self.settlement_candidate.qualification_attempt_digest
                != self.attempt_artifact_sha256
                or self.settlement_candidate.qualification_report_digest
                != self.report_digest
            ):
                raise QualificationIntakeError(
                    "settlement projection differs from qualification outcome"
                )


@dataclass(frozen=True)
class QualificationIntakeBatch:
    authority_manifest_digest: str
    outcomes: tuple[QualificationIntakeOutcome, ...]
    attempt_ref: EvidenceArtifactRef | None = None
    retry_plan: QualificationRetryPlan | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authority_manifest_digest",
            _digest(self.authority_manifest_digest, "authority manifest"),
        )
        outcomes = tuple(self.outcomes)
        retry_ids = (
            tuple(
                reservation
                for group in self.retry_plan.reservation_groups
                for reservation in group
            )
            if self.retry_plan is not None
            else ()
        )
        no_decision_ids = tuple(
            row.reservation_digest
            for row in outcomes
            if row.decision is QualificationDecision.NO_DECISION
        )
        if (
            not outcomes
            or any(type(row) is not QualificationIntakeOutcome for row in outcomes)
            or any(row.authority_manifest_digest != self.authority_manifest_digest for row in outcomes)
            or len({row.reservation_digest for row in outcomes}) != len(outcomes)
            or (self.attempt_ref is not None and type(self.attempt_ref) is not EvidenceArtifactRef)
            or (self.retry_plan is not None and type(self.retry_plan) is not QualificationRetryPlan)
            or (
                self.retry_plan is not None
                and self.retry_plan.authority_manifest_digest
                != self.authority_manifest_digest
            )
            or (
                self.attempt_ref is None
                and any(row.attempt_artifact_sha256 is not None for row in outcomes)
            )
            or (
                self.attempt_ref is not None
                and any(
                    row.attempt_artifact_sha256 != self.attempt_ref.sha256
                    for row in outcomes
                )
            )
            or retry_ids != no_decision_ids
        ):
            raise QualificationIntakeError("qualification batch is internally inconsistent")
        object.__setattr__(self, "outcomes", outcomes)


def _failure_digest(manifest: QualificationAuthorityManifest, exc: BaseException) -> str:
    return canonical_digest(
        "optima.qualification.intake-failure",
        {
            "authority_manifest_digest": manifest.digest,
            "exception": type(exc).__name__,
            "message": str(exc)[:4096],
        },
    )


def _retry_plan(
    manifest: QualificationAuthorityManifest,
    reservations: tuple[QualificationReservation, ...],
    failure_digest: str,
    *,
    bisect: bool,
) -> QualificationRetryPlan:
    ids = tuple(row.reservation_digest for row in reservations)
    if bisect and len(ids) > 1:
        midpoint = len(ids) // 2
        groups = (ids[:midpoint], ids[midpoint:])
        strategy = "bisect"
    else:
        groups = tuple((row,) for row in ids)
        strategy = "requeue"
    return QualificationRetryPlan(manifest.digest, strategy, groups, failure_digest)


def _no_decision_batch(
    manifest: QualificationAuthorityManifest,
    exc: BaseException,
    *,
    reason: str,
) -> QualificationIntakeBatch:
    failure = _failure_digest(manifest, exc)
    outcomes = tuple(
        QualificationIntakeOutcome(
            row.reservation_digest,
            row.selected_delta_digest,
            manifest.digest,
            QualificationDecision.NO_DECISION,
            reason,
            True,
            failure_digest=failure,
        )
        for row in manifest.reservations
    )
    return QualificationIntakeBatch(
        manifest.digest,
        outcomes,
        retry_plan=_retry_plan(
            manifest,
            manifest.reservations,
            failure,
            bisect=manifest.lane == "registered",
        ),
    )


def run_qualification_intake(
    factory: QualificationPlanFactory,
    *,
    executor,
    entropy_provider,
    hidden_judge,
    deadline: float,
) -> QualificationIntakeBatch:
    """Run, reopen, and project one finalized cohort without settlement authority."""

    if type(factory) is not QualificationPlanFactory:
        raise QualificationIntakeError("qualification factory is not exactly typed")
    manifest = factory.manifest
    try:
        value = factory.build()
    except (QualificationIntakeError, QualificationRunnerError, OSError) as exc:
        return _no_decision_batch(manifest, exc, reason="qualification_plan")
    try:
        reference = run_causal_qualification(
            value,
            executor=executor,
            entropy_provider=entropy_provider,
            hidden_judge=hidden_judge,
            deadline=deadline,
        )
        if type(reference) is not EvidenceArtifactRef:
            raise QualificationIntakeError("qualification runner returned no typed artifact")
        attempt = reopen_causal_qualification(
            value.evidence_root, reference, expected=value
        )
    except RawSpeedEvidenceError as exc:
        return _no_decision_batch(manifest, exc, reason="raw_speed_evidence")
    except QualificationRunnerError as exc:
        return _no_decision_batch(manifest, exc, reason="qualification_runner")

    expected_attempt_type = (
        DiscoveryQualificationAttempt
        if manifest.lane == "discovery"
        else CohortQualificationAttempt
    )
    if type(attempt) is not expected_attempt_type:
        raise QualificationIntakeError("qualification attempt lane differs")
    if (
        attempt.authority_digest != manifest.authority_digest
        or attempt.source_digest != manifest.source_digest
        or len(attempt.reports) != len(manifest.reservations)
    ):
        raise QualificationIntakeError("qualification attempt differs from intake authority")
    expected_report_type = (
        DiscoveryCandidateQualificationReport
        if manifest.lane == "discovery"
        else CandidateQualificationReport
    )
    outcomes = []
    retry_reservations = []
    from optima.eval.marginal_runtime import PreparedMarginalRuntime

    prepared_candidates = (
        value.prepared.candidates
        if type(value.prepared) is PreparedMarginalRuntime
        else ()
    )
    for index, (reservation, report) in enumerate(
        zip(manifest.reservations, attempt.reports, strict=True)
    ):
        if (
            type(report) is not expected_report_type
            or report.selected_delta_digest != reservation.selected_delta_digest
        ):
            raise QualificationIntakeError("qualification report order differs")
        settlement_candidate = None
        if (
            report.decision is QualificationDecision.PASS
            and prepared_candidates
        ):
            from optima.settlement import SettlementCandidate

            settlement_candidate = SettlementCandidate.from_qualification(
                reservation_digest=reservation.reservation_digest,
                finalized_block=reservation.finalized_block,
                event_index=reservation.finalized_event_index,
                event_subindex=reservation.finalized_event_subindex,
                hotkey=reservation.hotkey,
                target_id=reservation.target_id,
                members=reservation.target_members,
                prepared=prepared_candidates[index],
                report=report,
                authority=manifest,
                attempt_ref=reference,
            )
        outcomes.append(
            QualificationIntakeOutcome(
                reservation.reservation_digest,
                reservation.selected_delta_digest,
                manifest.digest,
                report.decision,
                report.reason,
                report.retryable,
                attempt_artifact_sha256=reference.sha256,
                report_digest=report.digest,
                settlement_candidate=settlement_candidate,
            )
        )
        if report.decision is QualificationDecision.NO_DECISION:
            retry_reservations.append(reservation)
    retry_plan = None
    if retry_reservations:
        retry_failure = canonical_digest(
            "optima.qualification.intake-report-retry",
            {
                "attempt": reference.sha256,
                "authority_manifest_digest": manifest.digest,
                "reservations": [row.reservation_digest for row in retry_reservations],
            },
        )
        retry_plan = _retry_plan(
            manifest, tuple(retry_reservations), retry_failure, bisect=False
        )
    return QualificationIntakeBatch(
        manifest.digest, tuple(outcomes), reference, retry_plan
    )


__all__ = [
    "GraphEvidenceProduct",
    "GraphMemberObservation",
    "GraphShapeObservation",
    "GraphVariantObservation",
    "GraphVerificationObservation",
    "QualificationAuthorityManifest",
    "QualificationIntakeBatch",
    "QualificationIntakeError",
    "QualificationIntakeOutcome",
    "QualificationPlanFactory",
    "QualificationReservation",
    "QualificationRetryPlan",
    "publish_graph_observation",
    "run_qualification_intake",
]
