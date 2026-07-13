"""Pure, evidence-bound planning for settlement and evaluation-stack adoption.

This module deliberately has no persistence, chain, wallet, or qualification-grading
authority.  It accepts a projection produced only after retained qualification evidence
has been reopened, selects deterministically over one frozen incumbent, and emits an
append-only event plan for the transactional control-plane store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Iterable

from optima.eval.evidence_store import EvidenceArtifactRef
from optima.stack_identity import canonical_digest, require_sha256_hex
from optima.stack_manifest import EvaluationStackManifest
from optima.stack_plan import StackArmIdentity


_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_LANES = frozenset({"registered", "discovery"})


class SettlementError(ValueError):
    """A settlement projection or transition plan is not closed and canonical."""


def _digest(value: object, field: str, *, optional: bool = False) -> str:
    if optional and value == "":
        return ""
    try:
        return require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise SettlementError(str(exc)) from None


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise SettlementError(f"{field} is not a canonical identifier")
    return value


def _integer(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise SettlementError(f"{field} must be a nonnegative integer")
    return value


def _speedup(value: object) -> str:
    if not isinstance(value, str):
        raise SettlementError("speedup must be an exact decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        raise SettlementError("speedup is not decimal") from None
    if not parsed.is_finite() or parsed <= 1:
        raise SettlementError("a settlement candidate must have speedup greater than one")
    canonical = format(parsed.normalize(), "f")
    if "." in canonical:
        canonical = canonical.rstrip("0").rstrip(".")
    if value != canonical:
        raise SettlementError(f"speedup must use canonical decimal spelling {canonical!r}")
    return canonical


@dataclass(frozen=True)
class SettlementReproductionIdentity:
    """Exact contribution identity that an independent PASS must reproduce."""

    arena_digest: str
    target_id: str
    selected_delta_digest: str
    hotkey: str
    incumbent_stack_digest: str
    incumbent_tree_digest: str
    candidate_stack_digest: str
    candidate_tree_digest: str

    def __post_init__(self) -> None:
        for field in (
            "arena_digest", "selected_delta_digest", "incumbent_stack_digest",
            "incumbent_tree_digest", "candidate_stack_digest", "candidate_tree_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        for field in ("target_id", "hotkey"):
            object.__setattr__(self, field, _identifier(getattr(self, field), field))

    def to_dict(self) -> dict[str, str]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.settlement.reproduction-identity", self.to_dict())


@dataclass(frozen=True)
class SettlementQualification:
    """One independently selected, reopened, and regraded PASS projection.

    A qualification is deliberately not a settlement candidate.  Settlement authority
    exists only after two distinct qualifications reproduce the same contribution.
    """

    lane: str
    arena_digest: str
    reservation_digest: str
    finalized_block: int
    event_index: int
    event_subindex: int
    hotkey: str
    target_id: str
    members: tuple[str, ...]
    selected_delta_digest: str
    qualification_authority_digest: str
    qualification_plan_digest: str
    qualification_attempt_digest: str
    qualification_report_digest: str
    selection_commitment_digest: str
    selection_secret_commitment_digest: str
    selection_evidence_digest: str
    arm_digest: str
    incumbent_stack_digest: str
    incumbent_tree_digest: str
    candidate_stack_digest: str
    candidate_tree_digest: str
    speedup: str
    incumbent_manifest: EvaluationStackManifest
    proposal_digest: str = ""
    candidate_manifest: EvaluationStackManifest | None = None

    def __post_init__(self) -> None:
        if self.lane not in _LANES:
            raise SettlementError("settlement lane is unsupported")
        for field in (
            "arena_digest",
            "reservation_digest",
            "selected_delta_digest",
            "qualification_authority_digest",
            "qualification_plan_digest",
            "qualification_attempt_digest",
            "qualification_report_digest",
            "selection_commitment_digest",
            "selection_secret_commitment_digest",
            "selection_evidence_digest",
            "arm_digest",
            "incumbent_stack_digest",
            "incumbent_tree_digest",
            "candidate_stack_digest",
            "candidate_tree_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        for field in ("finalized_block", "event_index", "event_subindex"):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        for field in ("hotkey", "target_id"):
            object.__setattr__(self, field, _identifier(getattr(self, field), field))
        members = tuple(self.members)
        if (
            not members
            or members != tuple(sorted(set(members)))
            or any(_identifier(row, "member") != row for row in members)
        ):
            raise SettlementError("settlement members are not canonical")
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "speedup", _speedup(self.speedup))
        if (
            type(self.incumbent_manifest) is not EvaluationStackManifest
            or self.incumbent_manifest.digest != self.incumbent_stack_digest
            or self.incumbent_manifest.arena_digest != self.arena_digest
        ):
            raise SettlementError("incumbent manifest differs from candidate authority")
        if self.incumbent_stack_digest == self.candidate_stack_digest:
            raise SettlementError("candidate does not change the incumbent stack identity")
        if self.incumbent_tree_digest == self.candidate_tree_digest:
            raise SettlementError("candidate does not change the incumbent tree identity")
        if self.lane == "registered":
            if type(self.candidate_manifest) is not EvaluationStackManifest:
                raise SettlementError("registered candidate lacks its exact stack manifest")
            if self.candidate_manifest.digest != self.candidate_stack_digest:
                raise SettlementError("candidate manifest differs from candidate stack")
            replacement = self.candidate_manifest.entries.get(self.target_id)
            if (
                replacement is None
                or replacement.selected_delta_digest != self.selected_delta_digest
            ):
                raise SettlementError(
                    "candidate manifest differs from its target/delta projection"
                )
            if self.proposal_digest:
                raise SettlementError(
                    "registered candidates cannot name a discovery proposal"
                )
        elif self.candidate_manifest is not None:
            raise SettlementError("discovery candidates cannot install a stack manifest")
        elif not self.proposal_digest:
            raise SettlementError("discovery candidate lacks its proposal identity")
        object.__setattr__(
            self,
            "proposal_digest",
            _digest(
                self.proposal_digest,
                "proposal_digest",
                optional=self.lane == "registered",
            ),
        )
        if (
            self.candidate_manifest is not None
            and self.candidate_manifest.arena_digest != self.arena_digest
        ):
            raise SettlementError("candidate manifest belongs to another arena")

    @property
    def finalized_order(self) -> tuple[int, int, int, str]:
        return (
            self.finalized_block,
            self.event_index,
            self.event_subindex,
            self.reservation_digest,
        )

    @property
    def incumbent(self) -> StackArmIdentity:
        return StackArmIdentity(self.incumbent_stack_digest, self.incumbent_tree_digest)

    @property
    def challenger(self) -> StackArmIdentity:
        return StackArmIdentity(self.candidate_stack_digest, self.candidate_tree_digest)

    @property
    def reproduction_identity(self) -> SettlementReproductionIdentity:
        return SettlementReproductionIdentity(
            self.arena_digest,
            self.target_id,
            self.selected_delta_digest,
            self.hotkey,
            self.incumbent_stack_digest,
            self.incumbent_tree_digest,
            self.candidate_stack_digest,
            self.candidate_tree_digest,
        )

    @classmethod
    def from_qualification(
        cls,
        *,
        reservation_digest: str,
        finalized_block: int,
        event_index: int,
        event_subindex: int,
        hotkey: str,
        target_id: str,
        members: tuple[str, ...],
        prepared,
        report,
        authority,
        attempt_ref,
        attempt,
    ) -> "SettlementQualification":
        """Project already reopened trusted types without reimplementing their grader."""

        from optima.discovery import DiscoveryArmPlan
        from optima.eval.evidence_store import EvidenceArtifactRef
        from optima.eval.marginal_runtime import PreparedCandidateRuntime
        from optima.eval.qualification import QualificationDecision
        from optima.eval.qualification_intake import QualificationAuthorityManifest
        from optima.eval.qualification_runner import (
            CandidateQualificationReport,
            CohortQualificationAttempt,
            DiscoveryCandidateQualificationReport,
            DiscoveryQualificationAttempt,
        )
        from optima.stack_plan import MarginalArmPlan

        if type(prepared) is not PreparedCandidateRuntime:
            raise SettlementError("prepared candidate runtime is not exactly typed")
        if type(authority) is not QualificationAuthorityManifest:
            raise SettlementError("qualification authority is not exactly typed")
        if type(attempt_ref) is not EvidenceArtifactRef:
            raise SettlementError("qualification attempt reference is not exactly typed")
        if report.decision is not QualificationDecision.PASS:
            raise SettlementError("only an independently reopened PASS may settle")
        reservations = tuple(
            row for row in authority.reservations
            if row.reservation_digest == reservation_digest
        )
        if len(reservations) != 1:
            raise SettlementError("authority does not bind exactly one reservation")
        reservation = reservations[0]
        arm = prepared.arm
        if type(arm) is MarginalArmPlan:
            lane = "registered"
            if (
                type(report) is not CandidateQualificationReport
                or type(attempt) is not CohortQualificationAttempt
            ):
                raise SettlementError("registered candidate report has the wrong type")
            if (
                report.marginal_arm_digest != arm.digest
                or report.target_id != target_id
                or report.candidate_launch_digest != prepared.launch.digest
            ):
                raise SettlementError("registered report differs from prepared runtime")
            manifest = arm.candidate
        elif type(arm) is DiscoveryArmPlan:
            lane = "discovery"
            if (
                type(report) is not DiscoveryCandidateQualificationReport
                or type(attempt) is not DiscoveryQualificationAttempt
            ):
                raise SettlementError("discovery candidate report has the wrong type")
            if (
                report.discovery_arm_digest != arm.digest
                or report.candidate_launch_digest != prepared.launch.digest
            ):
                raise SettlementError("discovery report differs from prepared runtime")
            manifest = None
        else:  # pragma: no cover - PreparedCandidateRuntime already closes this union
            raise SettlementError("qualification arm is unsupported")
        if authority.lane != lane:
            raise SettlementError("qualification authority lane differs from its arm")
        if (
            attempt.authority_digest != authority.authority_digest
            or attempt.commitment.digest != authority.commitment_digest
            or sum(row.digest == report.digest for row in attempt.reports) != 1
        ):
            raise SettlementError("qualification attempt differs from its authority/report")
        if (
            reservation.selected_delta_digest != arm.selected_delta_digest
            or report.selected_delta_digest != arm.selected_delta_digest
            or authority.candidate_deltas.count(arm.selected_delta_digest) != 1
        ):
            raise SettlementError("qualification selected-delta identity differs")
        selection_evidence_digest = canonical_digest(
            "optima.settlement.selection-evidence",
            {
                "commitment_digest": attempt.commitment.digest,
                "entropy_digest": attempt.entropy.digest,
                "selection_digest": attempt.selection.digest,
            },
        )
        return cls(
            lane=lane,
            arena_digest=arm.incumbent.arena_digest,
            reservation_digest=reservation_digest,
            finalized_block=finalized_block,
            event_index=event_index,
            event_subindex=event_subindex,
            hotkey=hotkey,
            target_id=target_id,
            members=members,
            selected_delta_digest=arm.selected_delta_digest,
            qualification_authority_digest=authority.digest,
            qualification_plan_digest=authority.authority_digest,
            qualification_attempt_digest=attempt_ref.sha256,
            qualification_report_digest=report.digest,
            selection_commitment_digest=attempt.commitment.digest,
            selection_secret_commitment_digest=attempt.commitment.secret_commitment,
            selection_evidence_digest=selection_evidence_digest,
            arm_digest=arm.digest,
            incumbent_stack_digest=arm.baseline_before.stack_digest,
            incumbent_tree_digest=arm.baseline_before.tree_digest,
            candidate_stack_digest=arm.challenger.stack_digest,
            candidate_tree_digest=arm.challenger.tree_digest,
            speedup=report.speedup,
            incumbent_manifest=arm.incumbent,
            proposal_digest=(arm.proposal_digest if lane == "discovery" else ""),
            candidate_manifest=manifest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_digest": self.arena_digest,
            "arm_digest": self.arm_digest,
            "candidate_manifest": (
                None if self.candidate_manifest is None else self.candidate_manifest.to_dict()
            ),
            "candidate_stack_digest": self.candidate_stack_digest,
            "candidate_tree_digest": self.candidate_tree_digest,
            "event_index": self.event_index,
            "event_subindex": self.event_subindex,
            "finalized_block": self.finalized_block,
            "hotkey": self.hotkey,
            "incumbent_stack_digest": self.incumbent_stack_digest,
            "incumbent_tree_digest": self.incumbent_tree_digest,
            "incumbent_manifest": self.incumbent_manifest.to_dict(),
            "lane": self.lane,
            "members": list(self.members),
            "qualification_attempt_digest": self.qualification_attempt_digest,
            "qualification_authority_digest": self.qualification_authority_digest,
            "qualification_plan_digest": self.qualification_plan_digest,
            "qualification_report_digest": self.qualification_report_digest,
            "selection_commitment_digest": self.selection_commitment_digest,
            "selection_secret_commitment_digest": self.selection_secret_commitment_digest,
            "selection_evidence_digest": self.selection_evidence_digest,
            "proposal_digest": self.proposal_digest,
            "reservation_digest": self.reservation_digest,
            "selected_delta_digest": self.selected_delta_digest,
            "speedup": self.speedup,
            "target_id": self.target_id,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SettlementQualification":
        fields = set(cls.__dataclass_fields__)
        if type(value) is not dict or set(value) != fields:
            raise SettlementError("settlement qualification fields do not match")
        row = dict(value)
        members = row.get("members")
        if type(members) is not list:
            raise SettlementError("settlement qualification members are malformed")
        row["members"] = tuple(members)
        manifest = row.get("candidate_manifest")
        row["candidate_manifest"] = (
            None if manifest is None else EvaluationStackManifest.from_dict(manifest)
        )
        row["incumbent_manifest"] = EvaluationStackManifest.from_dict(
            row["incumbent_manifest"]
        )
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.settlement.qualification", self.to_dict())


@dataclass(frozen=True)
class SettlementCandidate:
    """A primary PASS and an independent reproduction of the exact same delta."""

    primary: SettlementQualification
    reproduction: SettlementQualification

    def __post_init__(self) -> None:
        if (
            type(self.primary) is not SettlementQualification
            or type(self.reproduction) is not SettlementQualification
        ):
            raise SettlementError("settlement candidate requires two exact qualifications")
        if self.primary.reproduction_identity != self.reproduction.reproduction_identity:
            raise SettlementError(
                "independent reproduction differs from the primary reproduction identity"
            )
        common = (
            "lane", "arena_digest", "reservation_digest", "finalized_block",
            "event_index", "event_subindex", "hotkey", "target_id", "members",
            "selected_delta_digest", "arm_digest", "incumbent_stack_digest",
            "incumbent_tree_digest", "candidate_stack_digest", "candidate_tree_digest",
            "incumbent_manifest", "proposal_digest", "candidate_manifest",
        )
        if any(
            getattr(self.primary, field) != getattr(self.reproduction, field)
            for field in common
        ):
            raise SettlementError(
                "independent reproduction differs from the primary contribution identity"
            )
        distinct = (
            "qualification_authority_digest", "qualification_plan_digest",
            "qualification_attempt_digest", "qualification_report_digest",
            "selection_commitment_digest", "selection_secret_commitment_digest",
            "selection_evidence_digest",
        )
        if any(
            getattr(self.primary, field) == getattr(self.reproduction, field)
            for field in distinct
        ):
            raise SettlementError(
                "independent reproduction reuses primary authority or evidence"
            )

    @classmethod
    def from_reproductions(
        cls,
        primary: SettlementQualification,
        reproduction: SettlementQualification,
    ) -> "SettlementCandidate":
        return cls(primary, reproduction)

    def __getattr__(self, field: str):
        # Keep common identity access explicit to the pair while callers migrate from
        # the former single-PASS candidate representation.
        if field in SettlementQualification.__dataclass_fields__:
            return getattr(self.primary, field)
        raise AttributeError(field)

    @property
    def speedup(self) -> str:
        """Conservative reproduced speed: the slower independently passing run."""

        return min(
            (self.primary.speedup, self.reproduction.speedup),
            key=Decimal,
        )

    @property
    def finalized_order(self) -> tuple[int, int, int, str]:
        return self.primary.finalized_order

    @property
    def incumbent(self) -> StackArmIdentity:
        return self.primary.incumbent

    @property
    def challenger(self) -> StackArmIdentity:
        return self.primary.challenger

    @property
    def reproduction_identity(self) -> SettlementReproductionIdentity:
        return self.primary.reproduction_identity

    def to_dict(self) -> dict[str, object]:
        return {
            "primary": self.primary.to_dict(),
            "reproduction": self.reproduction.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> "SettlementCandidate":
        if type(value) is not dict or set(value) != {"primary", "reproduction"}:
            raise SettlementError("settlement candidate fields do not match")
        return cls(
            SettlementQualification.from_dict(value["primary"]),
            SettlementQualification.from_dict(value["reproduction"]),
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.settlement.candidate.v2", self.to_dict())


@dataclass(frozen=True)
class SettlementEvidence:
    """Receipt that both retained attempt artifacts were reopened for one candidate."""

    candidate_digest: str
    reservation_digest: str
    primary_authority_digest: str
    primary_attempt_ref: EvidenceArtifactRef
    primary_report_digest: str
    primary_selection_evidence_digest: str
    reproduction_authority_digest: str
    reproduction_attempt_ref: EvidenceArtifactRef
    reproduction_report_digest: str
    reproduction_selection_evidence_digest: str

    def __post_init__(self) -> None:
        for field in (
            "candidate_digest", "reservation_digest", "primary_authority_digest",
            "primary_report_digest", "primary_selection_evidence_digest",
            "reproduction_authority_digest", "reproduction_report_digest",
            "reproduction_selection_evidence_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if (
            type(self.primary_attempt_ref) is not EvidenceArtifactRef
            or type(self.reproduction_attempt_ref) is not EvidenceArtifactRef
        ):
            raise SettlementError("settlement attempt references are not exactly typed")
        if any(
            left == right
            for left, right in (
                (self.primary_authority_digest, self.reproduction_authority_digest),
                (self.primary_attempt_ref.sha256, self.reproduction_attempt_ref.sha256),
                (self.primary_report_digest, self.reproduction_report_digest),
                (
                    self.primary_selection_evidence_digest,
                    self.reproduction_selection_evidence_digest,
                ),
            )
        ):
            raise SettlementError("settlement evidence does not contain a reproduction")

    @classmethod
    def bind(
        cls,
        candidate: SettlementCandidate,
        *,
        primary_attempt_ref: EvidenceArtifactRef,
        reproduction_attempt_ref: EvidenceArtifactRef,
    ) -> "SettlementEvidence":
        if type(candidate) is not SettlementCandidate:
            raise SettlementError("settlement evidence candidate is not exactly typed")
        if (
            type(primary_attempt_ref) is not EvidenceArtifactRef
            or primary_attempt_ref.sha256 != candidate.primary.qualification_attempt_digest
            or type(reproduction_attempt_ref) is not EvidenceArtifactRef
            or reproduction_attempt_ref.sha256
            != candidate.reproduction.qualification_attempt_digest
        ):
            raise SettlementError("settlement attempt references differ from the candidate")
        return cls(
            candidate.digest,
            candidate.reservation_digest,
            candidate.primary.qualification_authority_digest,
            primary_attempt_ref,
            candidate.primary.qualification_report_digest,
            candidate.primary.selection_evidence_digest,
            candidate.reproduction.qualification_authority_digest,
            reproduction_attempt_ref,
            candidate.reproduction.qualification_report_digest,
            candidate.reproduction.selection_evidence_digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_digest": self.candidate_digest,
            "primary_attempt_ref": self.primary_attempt_ref.to_dict(),
            "primary_authority_digest": self.primary_authority_digest,
            "primary_report_digest": self.primary_report_digest,
            "primary_selection_evidence_digest": self.primary_selection_evidence_digest,
            "reproduction_attempt_ref": self.reproduction_attempt_ref.to_dict(),
            "reproduction_authority_digest": self.reproduction_authority_digest,
            "reproduction_report_digest": self.reproduction_report_digest,
            "reproduction_selection_evidence_digest": self.reproduction_selection_evidence_digest,
            "reservation_digest": self.reservation_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SettlementEvidence":
        fields = set(cls.__dataclass_fields__)
        if type(value) is not dict or set(value) != fields:
            raise SettlementError("settlement evidence fields do not match")
        return cls(
            value["candidate_digest"],  # type: ignore[arg-type]
            value["reservation_digest"],  # type: ignore[arg-type]
            value["primary_authority_digest"],  # type: ignore[arg-type]
            EvidenceArtifactRef.from_dict(value["primary_attempt_ref"]),
            value["primary_report_digest"],  # type: ignore[arg-type]
            value["primary_selection_evidence_digest"],  # type: ignore[arg-type]
            value["reproduction_authority_digest"],  # type: ignore[arg-type]
            EvidenceArtifactRef.from_dict(value["reproduction_attempt_ref"]),
            value["reproduction_report_digest"],  # type: ignore[arg-type]
            value["reproduction_selection_evidence_digest"],  # type: ignore[arg-type]
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.settlement.evidence", self.to_dict())


class SettlementEventType(str, Enum):
    HOLD = "HOLD"
    CROWN = "CROWN"
    ADOPTION = "ADOPTION"
    RETIREMENT = "RETIREMENT"
    NEUTRALIZATION = "NEUTRALIZATION"
    STACK_TRANSITION = "STACK_TRANSITION"
    DISCOVERY_BOUNTY = "DISCOVERY_BOUNTY"


@dataclass(frozen=True)
class SettlementEvent:
    event_type: SettlementEventType
    sequence: int
    previous_event_digest: str
    candidate_digest: str
    subject_digest: str
    target_id: str
    from_stack_digest: str
    from_tree_digest: str
    to_stack_digest: str
    to_tree_digest: str
    reason: str

    def __post_init__(self) -> None:
        if type(self.event_type) is not SettlementEventType:
            raise SettlementError("settlement event type is not exact")
        object.__setattr__(self, "sequence", _integer(self.sequence, "event sequence"))
        for field in (
            "candidate_digest", "subject_digest", "from_stack_digest",
            "from_tree_digest", "to_stack_digest", "to_tree_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(
            self, "previous_event_digest",
            _digest(self.previous_event_digest, "previous_event_digest", optional=True),
        )
        for field in ("target_id", "reason"):
            object.__setattr__(self, field, _identifier(getattr(self, field), field))

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_digest": self.candidate_digest,
            "event_type": self.event_type.value,
            "from_stack_digest": self.from_stack_digest,
            "from_tree_digest": self.from_tree_digest,
            "previous_event_digest": self.previous_event_digest,
            "reason": self.reason,
            "sequence": self.sequence,
            "subject_digest": self.subject_digest,
            "target_id": self.target_id,
            "to_stack_digest": self.to_stack_digest,
            "to_tree_digest": self.to_tree_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> "SettlementEvent":
        fields = set(cls.__dataclass_fields__)
        if type(value) is not dict or set(value) != fields:
            raise SettlementError("settlement event fields do not match")
        row = dict(value)
        try:
            row["event_type"] = SettlementEventType(row["event_type"])
        except (TypeError, ValueError):
            raise SettlementError("settlement event type is unsupported") from None
        return cls(**row)  # type: ignore[arg-type]

    @property
    def digest(self) -> str:
        return canonical_digest("optima.settlement.event", self.to_dict())


@dataclass(frozen=True)
class StackTransitionOutput:
    candidate_digest: str
    before: StackArmIdentity
    after: StackArmIdentity
    manifest: EvaluationStackManifest

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_digest", _digest(self.candidate_digest, "candidate"))
        if (
            type(self.before) is not StackArmIdentity
            or type(self.after) is not StackArmIdentity
            or type(self.manifest) is not EvaluationStackManifest
            or self.manifest.digest != self.after.stack_digest
            or self.before == self.after
        ):
            raise SettlementError("stack transition output is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return {
            "after": self.after.to_dict(),
            "before": self.before.to_dict(),
            "candidate_digest": self.candidate_digest,
            "manifest": self.manifest.to_dict(),
        }


@dataclass(frozen=True)
class SettlementPlan:
    before: StackArmIdentity
    after: StackArmIdentity
    winner_candidate_digest: str
    events: tuple[SettlementEvent, ...]
    transition: StackTransitionOutput | None

    def __post_init__(self) -> None:
        if type(self.before) is not StackArmIdentity or type(self.after) is not StackArmIdentity:
            raise SettlementError("settlement plan arms are not typed")
        object.__setattr__(
            self, "winner_candidate_digest",
            _digest(self.winner_candidate_digest, "winner", optional=True),
        )
        events = tuple(self.events)
        if any(type(row) is not SettlementEvent for row in events):
            raise SettlementError("settlement events are not typed")
        for prior, current in zip(events, events[1:]):
            if current.sequence != prior.sequence + 1 or current.previous_event_digest != prior.digest:
                raise SettlementError("settlement event journal is not contiguous")
        if self.transition is None:
            if self.winner_candidate_digest or self.before != self.after:
                raise SettlementError("non-transition settlement plan names a winner")
        elif (
            type(self.transition) is not StackTransitionOutput
            or self.transition.candidate_digest != self.winner_candidate_digest
            or self.transition.before != self.before
            or self.transition.after != self.after
        ):
            raise SettlementError("settlement transition differs from its plan")
        object.__setattr__(self, "events", events)

    def to_dict(self) -> dict[str, object]:
        return {
            "after": self.after.to_dict(),
            "before": self.before.to_dict(),
            "events": [row.to_dict() for row in self.events],
            "transition": None if self.transition is None else self.transition.to_dict(),
            "winner_candidate_digest": self.winner_candidate_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.settlement.plan", self.to_dict())


class _Journal:
    def __init__(self, sequence: int, previous: str):
        self.sequence = _integer(sequence, "initial event sequence")
        self.previous = _digest(previous, "previous event", optional=True)
        self.events: list[SettlementEvent] = []

    def add(
        self,
        event_type: SettlementEventType,
        candidate: SettlementCandidate,
        *,
        subject_digest: str,
        target_id: str,
        before: StackArmIdentity,
        after: StackArmIdentity,
        reason: str,
    ) -> None:
        event = SettlementEvent(
            event_type, self.sequence, self.previous, candidate.digest,
            subject_digest, target_id, before.stack_digest, before.tree_digest,
            after.stack_digest, after.tree_digest, reason,
        )
        self.events.append(event)
        self.previous = event.digest
        self.sequence += 1


def plan_settlement(
    candidates: Iterable[SettlementCandidate],
    *,
    current_manifest: EvaluationStackManifest,
    current_tree_digest: str,
    initial_event_sequence: int = 0,
    previous_event_digest: str = "",
) -> SettlementPlan:
    """Select one registered winner over one incumbent and emit a hash-chained plan."""

    if type(current_manifest) is not EvaluationStackManifest:
        raise SettlementError("current manifest is not exactly typed")
    before = StackArmIdentity(current_manifest.digest, current_tree_digest)
    rows = tuple(candidates)
    if any(type(row) is not SettlementCandidate for row in rows):
        raise SettlementError("settlement candidates are not exactly typed")
    if len({row.digest for row in rows}) != len(rows) or len(
        {row.reservation_digest for row in rows}
    ) != len(rows):
        raise SettlementError("settlement candidates contain duplicates")
    journal = _Journal(initial_event_sequence, previous_event_digest)

    current = tuple(row for row in rows if row.incumbent == before)
    stale = sorted((row for row in rows if row.incumbent != before), key=lambda row: row.finalized_order)
    for row in stale:
        journal.add(
            SettlementEventType.HOLD, row, subject_digest=row.selected_delta_digest,
            target_id=row.target_id, before=before, after=before,
            reason="stale_incumbent",
        )

    discoveries = sorted(
        (row for row in current if row.lane == "discovery"),
        key=lambda row: row.finalized_order,
    )
    for row in discoveries:
        journal.add(
            SettlementEventType.DISCOVERY_BOUNTY, row,
            subject_digest=row.selected_delta_digest, target_id=row.target_id,
            before=before, after=before, reason="qualified_discovery",
        )

    registered = tuple(row for row in current if row.lane == "registered")
    if not registered:
        return SettlementPlan(before, before, "", tuple(journal.events), None)
    winner = min(
        registered,
        key=lambda row: (-Decimal(row.speedup), row.finalized_order),
    )
    after = winner.challenger
    for row in sorted(
        (item for item in registered if item is not winner),
        key=lambda item: item.finalized_order,
    ):
        reason = (
            "conflict_lost"
            if set(row.members) & set(winner.members)
            else "incumbent_advanced"
        )
        journal.add(
            SettlementEventType.HOLD, row, subject_digest=row.selected_delta_digest,
            target_id=row.target_id, before=before, after=before, reason=reason,
        )

    assert winner.candidate_manifest is not None
    replacement = winner.candidate_manifest.entries[winner.target_id]
    journal.add(
        SettlementEventType.CROWN, winner, subject_digest=replacement.digest,
        target_id=winner.target_id, before=before, after=before, reason="qualified_win",
    )
    prior = current_manifest.entries.get(winner.target_id)
    if prior is not None:
        journal.add(
            SettlementEventType.RETIREMENT, winner, subject_digest=prior.digest,
            target_id=prior.target_id, before=before, after=before, reason="superseded",
        )
    for target_id, ref in sorted(current_manifest.entries.items()):
        if target_id != winner.target_id and target_id not in winner.candidate_manifest.entries:
            journal.add(
                SettlementEventType.NEUTRALIZATION, winner, subject_digest=ref.digest,
                target_id=target_id, before=before, after=before, reason="displaced",
            )
    journal.add(
        SettlementEventType.ADOPTION, winner, subject_digest=replacement.digest,
        target_id=winner.target_id, before=before, after=after, reason="adopted",
    )
    journal.add(
        SettlementEventType.STACK_TRANSITION, winner, subject_digest=winner.candidate_stack_digest,
        target_id=winner.target_id, before=before, after=after, reason="incumbent_updated",
    )
    transition = StackTransitionOutput(winner.digest, before, after, winner.candidate_manifest)
    return SettlementPlan(
        before, after, winner.digest, tuple(journal.events), transition
    )


__all__ = [
    "SettlementCandidate", "SettlementError", "SettlementEvidence", "SettlementEvent",
    "SettlementEventType", "SettlementPlan", "SettlementQualification",
    "SettlementReproductionIdentity", "StackTransitionOutput", "plan_settlement",
]
