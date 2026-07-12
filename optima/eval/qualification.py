"""Content-addressed graph-verification veto for later qualification.

This pure module cannot score, crown, settle, or grade model quality.  It only
recomputes whether one exact marginal candidate has complete eager and graph
replay evidence.  PASS is necessary for later qualification, never sufficient.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import ClassVar

from optima.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


GRAPH_QUALIFICATION_SCHEMA_VERSION = 1
GRAPH_QUALIFICATION_POLICY_VERSION = "graph-verification-veto.v1"
GRAPH_EVIDENCE_DOMAIN = "qualification.graph-verification"
GRAPH_EVIDENCE_MEDIA_TYPE = "application/vnd.optima.graph-verification+json"
GRAPH_EVIDENCE_SCHEMA = "optima.qualification.graph-raw-evidence.v1"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_FAILURES = frozenset(
    {"none", "not_applicable", "eager", "capture", "replay", "graph_not_required"}
)


class QualificationError(ValueError):
    """Graph-verification policy or evidence is malformed."""


class QualificationDecision(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NO_DECISION = "NO_DECISION"


def _digest(value: object, field: str) -> str:
    try:
        result = require_sha256_hex(value, field=field)
    except StackIdentityError as exc:
        raise QualificationError(str(exc)) from exc
    if result == "0" * 64:
        raise QualificationError(f"{field} must not be the all-zero digest")
    return result


def _id(value: object, field: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise QualificationError(f"{field} must be a canonical identifier")
    return value


def _integer(value: object, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise QualificationError(f"{field} must be an integer >= {minimum}")
    return value


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise QualificationError(f"{field} must be boolean")
    return value


def _array(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise QualificationError(f"{label} must be an array")
    return value


def _strict(value: object, expected: frozenset[str], label: str) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or not all(isinstance(key, str) for key in value)
        or frozenset(value) != expected
    ):
        raise QualificationError(f"{label} fields do not match the schema")
    return value


def _encode(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _encode(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if value is None or type(value) in {str, bool, int}:
        return value
    raise QualificationError(f"canonical record contains unsupported {type(value).__name__}")


def _load(cls, value: object, label: str, **converters):
    names = frozenset(field.name for field in fields(cls))
    raw = _strict(value, names, label)
    return cls(
        **{
            name: converters[name](raw[name]) if name in converters else raw[name]
            for name in names
        }
    )


def _ordered(rows: tuple[object, ...], keys: tuple[object, ...], label: str) -> None:
    if not rows:
        raise QualificationError(f"{label} must not be empty")
    if len(set(keys)) != len(keys):
        raise QualificationError(f"{label} contains duplicates")
    if keys != tuple(sorted(keys)):
        raise QualificationError(f"{label} must be canonically ordered")


def _version(policy: object, schema: object, label: str) -> None:
    if (
        policy != GRAPH_QUALIFICATION_POLICY_VERSION
        or type(schema) is not int
        or schema != GRAPH_QUALIFICATION_SCHEMA_VERSION
    ):
        raise QualificationError(f"unsupported {label} policy/schema")


class _Canonical:
    _domain: ClassVar[str]

    def to_dict(self) -> dict[str, object]:
        encoded = _encode(self)
        assert isinstance(encoded, dict)
        return encoded

    @property
    def digest(self) -> str:
        return canonical_digest(self._domain, self.to_dict())


@dataclass(frozen=True)
class GraphVerificationMemberBinding(_Canonical):
    _domain: ClassVar[str] = "optima.qualification.graph-member-binding"
    slot_id: str
    target_spec_digest: str
    contract_digest: str
    verification_profile_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id", _id(self.slot_id, "member slot_id"))
        for field in ("target_spec_digest", "contract_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), f"member {field}"))
        object.__setattr__(
            self, "verification_profile_id", _id(self.verification_profile_id, "profile ID")
        )

    @classmethod
    def from_dict(cls, value: object) -> "GraphVerificationMemberBinding":
        return _load(cls, value, "graph member binding")


@dataclass(frozen=True)
class GraphVerificationBinding(_Canonical):
    _domain: ClassVar[str] = "optima.qualification.graph-binding"
    marginal_arm_digest: str
    candidate_launch_digest: str
    contribution_ref_digest: str
    selected_delta_digest: str
    target_id: str
    target_spec_digest: str
    catalog_digest: str
    members: tuple[GraphVerificationMemberBinding, ...]
    verification_policy_digest: str

    def __post_init__(self) -> None:
        for field in (
            "marginal_arm_digest",
            "candidate_launch_digest",
            "contribution_ref_digest",
            "selected_delta_digest",
            "target_spec_digest",
            "catalog_digest",
            "verification_policy_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "target_id", _id(self.target_id, "target_id"))
        members = tuple(self.members)
        if not all(type(row) is GraphVerificationMemberBinding for row in members):
            raise QualificationError("graph members must be typed bindings")
        _ordered(members, tuple(row.slot_id for row in members), "graph members")
        object.__setattr__(self, "members", members)

    @classmethod
    def from_dict(cls, value: object) -> "GraphVerificationBinding":
        return _load(
            cls,
            value,
            "graph verification binding",
            members=lambda rows: tuple(
                GraphVerificationMemberBinding.from_dict(row)
                for row in _array(rows, "graph members")
            ),
        )


@dataclass(frozen=True)
class GraphVariantRequirement(_Canonical):
    _domain: ClassVar[str] = "optima.qualification.graph-variant-requirement"
    slot_id: str
    variant_id: str
    shape_descriptor_digests: tuple[str, ...]
    context_applicable: bool
    applicable_shape_descriptor_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id", _id(self.slot_id, "variant slot_id"))
        object.__setattr__(self, "variant_id", _id(self.variant_id, "variant_id"))
        shapes = tuple(
            _digest(value, "shape descriptor digest")
            for value in self.shape_descriptor_digests
        )
        _ordered(shapes, shapes, "variant shape requirements")
        object.__setattr__(self, "shape_descriptor_digests", shapes)
        object.__setattr__(
            self, "context_applicable", _boolean(self.context_applicable, "context applicability")
        )
        applicable = tuple(
            _digest(value, "applicable shape digest")
            for value in self.applicable_shape_descriptor_digests
        )
        if applicable:
            _ordered(applicable, applicable, "applicable shape requirements")
        if not set(applicable) <= set(shapes) or bool(applicable) != self.context_applicable:
            raise QualificationError("required graph applicability is inconsistent")
        object.__setattr__(self, "applicable_shape_descriptor_digests", applicable)

    @classmethod
    def from_dict(cls, value: object) -> "GraphVariantRequirement":
        return _load(
            cls,
            value,
            "graph variant requirement",
            shape_descriptor_digests=lambda rows: tuple(_array(rows, "shape requirements")),
            applicable_shape_descriptor_digests=lambda rows: tuple(
                _array(rows, "applicable shape requirements")
            ),
        )


@dataclass(frozen=True)
class GraphVerificationRequirement(_Canonical):
    _domain: ClassVar[str] = "optima.qualification.graph-requirement"
    binding: GraphVerificationBinding
    variants: tuple[GraphVariantRequirement, ...]
    expected_graph_replays: int
    policy_version: str = GRAPH_QUALIFICATION_POLICY_VERSION
    schema_version: int = GRAPH_QUALIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.binding) is not GraphVerificationBinding:
            raise QualificationError("graph requirement binding is not typed")
        _version(self.policy_version, self.schema_version, "graph requirement")
        object.__setattr__(
            self, "expected_graph_replays", _integer(self.expected_graph_replays, "replays", 2)
        )
        variants = tuple(self.variants)
        if not all(type(row) is GraphVariantRequirement for row in variants):
            raise QualificationError("graph variants must be typed requirements")
        _ordered(
            variants,
            tuple((row.slot_id, row.variant_id) for row in variants),
            "graph variants",
        )
        if {row.slot_id for row in variants} != {row.slot_id for row in self.binding.members}:
            raise QualificationError("graph variants must cover every bound member exactly")
        object.__setattr__(self, "variants", variants)

    @classmethod
    def from_dict(cls, value: object) -> "GraphVerificationRequirement":
        return _load(
            cls,
            value,
            "graph verification requirement",
            binding=GraphVerificationBinding.from_dict,
            variants=lambda rows: tuple(
                GraphVariantRequirement.from_dict(row) for row in _array(rows, "graph variants")
            ),
        )


@dataclass(frozen=True)
class GraphShapeEvidence(_Canonical):
    _domain: ClassVar[str] = "optima.qualification.graph-shape-evidence"
    descriptor_digest: str
    applicable: bool
    eager_passed: bool
    graph_required: bool
    graph_replays: int
    graph_passed: bool
    failure_kind: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "descriptor_digest",
            _digest(self.descriptor_digest, "shape digest"),
        )
        for field in ("applicable", "eager_passed", "graph_required", "graph_passed"):
            object.__setattr__(self, field, _boolean(getattr(self, field), field))
        object.__setattr__(self, "graph_replays", _integer(self.graph_replays, "replays"))
        if self.failure_kind not in _FAILURES:
            raise QualificationError("shape failure_kind is unsupported")
        state = (self.applicable, self.eager_passed, self.graph_required, self.graph_passed)
        allowed = {
            "not_applicable": (False, False, False, False),
            "eager": (True, False, True, False),
            "capture": (True, True, True, False),
            "replay": (True, True, True, False),
            "graph_not_required": (True, True, False, False),
        }
        if self.failure_kind == "none":
            valid = state == (True, True, True, True) and self.graph_replays >= 1
        else:
            valid = state == allowed[self.failure_kind] and (
                self.failure_kind == "replay" or self.graph_replays == 0
            )
        if not valid:
            raise QualificationError("graph shape evidence is internally inconsistent")

    @classmethod
    def from_dict(cls, value: object) -> "GraphShapeEvidence":
        return _load(cls, value, "graph shape evidence")


@dataclass(frozen=True)
class GraphVariantEvidence(_Canonical):
    _domain: ClassVar[str] = "optima.qualification.graph-variant-evidence"
    slot_id: str
    variant_id: str
    context_applicable: bool
    domain_coverage_complete: bool
    shapes: tuple[GraphShapeEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id", _id(self.slot_id, "evidence slot_id"))
        object.__setattr__(self, "variant_id", _id(self.variant_id, "evidence variant_id"))
        for field in ("context_applicable", "domain_coverage_complete"):
            object.__setattr__(self, field, _boolean(getattr(self, field), field))
        shapes = tuple(self.shapes)
        if not all(type(row) is GraphShapeEvidence for row in shapes):
            raise QualificationError("variant shapes must be typed evidence")
        _ordered(shapes, tuple(row.descriptor_digest for row in shapes), "variant shapes")
        if not self.context_applicable and any(
            row.failure_kind != "not_applicable" for row in shapes
        ):
            raise QualificationError("context-inapplicable variant has executable evidence")
        object.__setattr__(self, "shapes", shapes)

    @classmethod
    def from_dict(cls, value: object) -> "GraphVariantEvidence":
        return _load(
            cls,
            value,
            "graph variant evidence",
            shapes=lambda rows: tuple(
                GraphShapeEvidence.from_dict(row) for row in _array(rows, "variant shapes")
            ),
        )


@dataclass(frozen=True)
class GraphMemberEvidence(_Canonical):
    _domain: ClassVar[str] = "optima.qualification.graph-member-evidence"
    slot_id: str
    variants: tuple[GraphVariantEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_id", _id(self.slot_id, "member evidence slot_id"))
        variants = tuple(self.variants)
        if not all(
            type(row) is GraphVariantEvidence and row.slot_id == self.slot_id
            for row in variants
        ):
            raise QualificationError("member variants are invalid or name another slot")
        _ordered(variants, tuple(row.variant_id for row in variants), "member variants")
        object.__setattr__(self, "variants", variants)

    @classmethod
    def from_dict(cls, value: object) -> "GraphMemberEvidence":
        return _load(
            cls,
            value,
            "graph member evidence",
            variants=lambda rows: tuple(
                GraphVariantEvidence.from_dict(row) for row in _array(rows, "member variants")
            ),
        )


@dataclass(frozen=True)
class GraphVerificationRawEvidence(_Canonical):
    _domain: ClassVar[str] = "optima.qualification.graph-raw-evidence"
    requirement_digest: str
    members: tuple[GraphMemberEvidence, ...]
    policy_version: str = GRAPH_QUALIFICATION_POLICY_VERSION
    schema_version: int = GRAPH_QUALIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requirement_digest",
            _digest(self.requirement_digest, "requirement"),
        )
        _version(self.policy_version, self.schema_version, "raw graph evidence")
        members = tuple(self.members)
        if not all(type(row) is GraphMemberEvidence for row in members):
            raise QualificationError("raw members must be typed evidence")
        _ordered(members, tuple(row.slot_id for row in members), "raw members")
        object.__setattr__(self, "members", members)

    @classmethod
    def from_dict(cls, value: object) -> "GraphVerificationRawEvidence":
        return _load(
            cls,
            value,
            "raw graph evidence",
            members=lambda rows: tuple(
                GraphMemberEvidence.from_dict(row) for row in _array(rows, "raw members")
            ),
        )


@dataclass(frozen=True)
class GraphVerificationEvidenceRef(_Canonical):
    _domain: ClassVar[str] = "optima.qualification.graph-evidence-ref"
    binding: GraphVerificationBinding
    requirement_digest: str
    raw_evidence_digest: str
    policy_version: str = GRAPH_QUALIFICATION_POLICY_VERSION
    schema_version: int = GRAPH_QUALIFICATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.binding) is not GraphVerificationBinding:
            raise QualificationError("graph evidence reference binding is not typed")
        for field in ("requirement_digest", "raw_evidence_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        _version(self.policy_version, self.schema_version, "graph evidence reference")

    @classmethod
    def from_dict(cls, value: object) -> "GraphVerificationEvidenceRef":
        return _load(
            cls,
            value,
            "graph evidence reference",
            binding=GraphVerificationBinding.from_dict,
        )


@dataclass(frozen=True)
class GraphVerificationGrade(_Canonical):
    _domain: ClassVar[str] = "optima.qualification.graph-grade"
    decision: QualificationDecision
    reason: str
    requirement_digest: str
    evidence_ref_digest: str | None
    raw_evidence_digest: str | None

    def __post_init__(self) -> None:
        if type(self.decision) is not QualificationDecision:
            raise QualificationError("graph grade decision is not typed")
        object.__setattr__(self, "reason", _id(self.reason, "grade reason"))
        object.__setattr__(
            self,
            "requirement_digest",
            _digest(self.requirement_digest, "requirement"),
        )
        for field in ("evidence_ref_digest", "raw_evidence_digest"):
            value = getattr(self, field)
            if value is not None:
                object.__setattr__(self, field, _digest(value, field))

    @property
    def veto_passed(self) -> bool:
        return self.decision is QualificationDecision.PASS

    @classmethod
    def from_dict(cls, value: object) -> "GraphVerificationGrade":
        def decision(item: object) -> QualificationDecision:
            try:
                return QualificationDecision(item)
            except (TypeError, ValueError) as exc:
                raise QualificationError("graph grade decision is unsupported") from exc

        return _load(cls, value, "graph grade", decision=decision)


def _grade(
    decision: QualificationDecision,
    reason: str,
    requirement: GraphVerificationRequirement,
    reference: GraphVerificationEvidenceRef | None,
    raw: GraphVerificationRawEvidence | None,
) -> GraphVerificationGrade:
    return GraphVerificationGrade(
        decision,
        reason,
        requirement.digest,
        None if reference is None else reference.digest,
        None if raw is None else raw.digest,
    )


def regrade_graph_verification(
    requirement: GraphVerificationRequirement,
    evidence_ref: GraphVerificationEvidenceRef | None,
    raw_evidence: GraphVerificationRawEvidence | None,
) -> GraphVerificationGrade:
    """Recompute the mandatory graph veto without trusting an aggregate boolean."""

    if type(requirement) is not GraphVerificationRequirement:
        raise QualificationError("graph requirement is not typed")
    def result(decision: QualificationDecision, reason: str) -> GraphVerificationGrade:
        return _grade(decision, reason, requirement, evidence_ref, raw_evidence)
    if evidence_ref is None or raw_evidence is None:
        return result(QualificationDecision.NO_DECISION, "graph_evidence_missing")
    if type(evidence_ref) is not GraphVerificationEvidenceRef or type(
        raw_evidence
    ) is not GraphVerificationRawEvidence:
        raise QualificationError("graph evidence is not typed")
    if (
        evidence_ref.binding != requirement.binding
        or evidence_ref.requirement_digest != requirement.digest
        or raw_evidence.requirement_digest != requirement.digest
    ):
        return result(QualificationDecision.NO_DECISION, "graph_identity_mismatch")
    if evidence_ref.raw_evidence_digest != raw_evidence.digest:
        return result(QualificationDecision.NO_DECISION, "graph_evidence_tampered")

    expected: dict[str, tuple[GraphVariantRequirement, ...]] = {}
    for variant in requirement.variants:
        expected[variant.slot_id] = expected.get(variant.slot_id, ()) + (variant,)
    observed = {member.slot_id: member for member in raw_evidence.members}
    member_ids = tuple(member.slot_id for member in requirement.binding.members)
    if tuple(observed) != member_ids:
        return result(QualificationDecision.NO_DECISION, "graph_member_coverage_incomplete")

    for slot_id in member_ids:
        required_variants, actual_variants = expected[slot_id], observed[slot_id].variants
        if tuple(row.variant_id for row in actual_variants) != tuple(
            row.variant_id for row in required_variants
        ):
            return result(QualificationDecision.NO_DECISION, "graph_variant_coverage_incomplete")
        applicable = 0
        for required, actual in zip(required_variants, actual_variants):
            if actual.slot_id != required.slot_id or tuple(
                row.descriptor_digest for row in actual.shapes
            ) != required.shape_descriptor_digests:
                return result(QualificationDecision.NO_DECISION, "graph_shape_coverage_incomplete")
            actual_applicable = tuple(
                shape.descriptor_digest for shape in actual.shapes if shape.applicable
            )
            if (
                actual.context_applicable != required.context_applicable
                or actual_applicable != required.applicable_shape_descriptor_digests
            ):
                return result(QualificationDecision.FAIL, "graph_applicability_failed")
            if actual.context_applicable and not actual.domain_coverage_complete:
                return result(QualificationDecision.FAIL, "graph_domain_coverage_failed")
            for shape in actual.shapes:
                if not shape.applicable:
                    continue
                applicable += 1
                if shape.failure_kind != "none":
                    return result(QualificationDecision.FAIL, f"graph_{shape.failure_kind}_failed")
                if shape.graph_replays != requirement.expected_graph_replays:
                    return result(QualificationDecision.NO_DECISION, "graph_replay_count_mismatch")
        if applicable == 0:
            return result(QualificationDecision.FAIL, "graph_member_not_applicable")
    return result(QualificationDecision.PASS, "graph_verification_pass")


def reopen_graph_verification(
    root: object,
    artifact_ref: object,
    requirement: GraphVerificationRequirement,
    evidence_ref: GraphVerificationEvidenceRef,
) -> GraphVerificationGrade:
    """Reopen controller-owned raw bytes and recompute the graph veto."""

    from optima.eval.evidence_store import (
        EvidenceArtifactRef,
        EvidenceStoreError,
        reopen_evidence,
    )

    if type(artifact_ref) is not EvidenceArtifactRef or (
        artifact_ref.domain != GRAPH_EVIDENCE_DOMAIN
        or artifact_ref.media_type != GRAPH_EVIDENCE_MEDIA_TYPE
        or artifact_ref.schema != GRAPH_EVIDENCE_SCHEMA
    ):
        raise QualificationError("graph evidence artifact reference is invalid")
    try:
        payload = reopen_evidence(root, artifact_ref)
    except EvidenceStoreError as exc:
        raise QualificationError(f"graph evidence artifact failed to reopen: {exc}") from None

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise QualificationError(f"graph evidence JSON repeats key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise QualificationError(f"graph evidence JSON contains {value}")

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=reject_constant,
        )
        if canonical_json_bytes(decoded) != payload:
            raise QualificationError("graph evidence JSON encoding is not canonical")
    except QualificationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, StackIdentityError) as exc:
        raise QualificationError(f"graph evidence JSON is invalid: {exc}") from None
    raw = GraphVerificationRawEvidence.from_dict(decoded)
    return regrade_graph_verification(requirement, evidence_ref, raw)


@dataclass(frozen=True)
class ReferenceManifest(_Canonical):
    """Exact candidate-free engine and hidden authority used by pristine T."""

    _domain: ClassVar[str] = "optima.qualification.reference-manifest"
    pristine_stack_digest: str
    pristine_tree_digest: str
    pristine_launch_digest: str
    runtime_digest: str
    base_engine_digest: str
    arena_digest: str
    catalog_digest: str
    controller_distribution_digest: str
    worker_distribution_digest: str
    model_revision_digest: str
    model_manifest_digest: str
    model_content_digest: str
    logical_hardware_digest: str
    workload_digest: str
    tokenizer_digest: str
    hidden_corpus_commitment: str
    hidden_judge_digest: str
    selection_policy_digest: str

    def __post_init__(self) -> None:
        for item in fields(self):
            object.__setattr__(self, item.name, _digest(getattr(self, item.name), item.name))

    @classmethod
    def from_pristine(
        cls,
        stack: object,
        launch: object,
        binding: object,
        *,
        workload_digest: str,
        tokenizer_digest: str,
        hidden_corpus_commitment: str,
        hidden_judge_digest: str,
        selection_policy_digest: str,
    ) -> "ReferenceManifest":
        from optima.eval.engine_launch import EngineLaunchSpec
        from optima.eval.marginal_runtime import (
            MaterializedArmBinding,
            _resolve_materialized_binding,
        )
        from optima.stack_manifest import EvaluationStackManifest

        if (
            type(stack) is not EvaluationStackManifest
            or type(launch) is not EngineLaunchSpec
            or type(binding) is not MaterializedArmBinding
        ):
            raise QualificationError("pristine reference stack/launch/binding is not typed")
        if stack.entries:
            raise QualificationError("pristine reference cannot contain proposal contributions")
        if (
            launch.stack_digest != stack.digest
            or (launch.runtime_digest, launch.base_engine_digest, launch.arena_digest)
            != (stack.runtime_digest, stack.base_engine_digest, stack.arena_digest)
        ):
            raise QualificationError("pristine reference launch differs from its empty stack")
        try:
            _resolve_materialized_binding(
                launch, binding, stack, expected_tree_digest=launch.tree_digest
            )
        except (OSError, TypeError, ValueError) as exc:
            raise QualificationError(f"pristine reference tree cannot reopen: {exc}") from None
        return cls(
            stack.digest,
            launch.tree_digest,
            launch.digest,
            launch.runtime_digest,
            launch.base_engine_digest,
            launch.arena_digest,
            stack.catalog_digest,
            launch.controller_distribution_digest,
            launch.worker_distribution_digest,
            launch.model_revision_digest,
            launch.model_manifest_digest,
            launch.model_content_digest,
            launch.hardware.digest,
            workload_digest,
            tokenizer_digest,
            hidden_corpus_commitment,
            hidden_judge_digest,
            selection_policy_digest,
        )

    @classmethod
    def from_dict(cls, value: object) -> "ReferenceManifest":
        return _load(cls, value, "reference manifest")


@dataclass(frozen=True)
class QualificationProfile(_Canonical):
    """Validator-owned policy; submissions cannot choose gates or calibration."""

    _domain: ClassVar[str] = "optima.qualification.profile"
    reference: ReferenceManifest
    calibration_context_digest: str
    calibration_digest: str
    graph_requirement_digest: str
    required_quality_metrics: tuple[str, ...]
    nll_tail_threshold: str
    tokens_per_prompt: int
    topk_width: int
    hidden_tasks_per_prompt: int
    support_policy_digest: str
    hidden_task_policy_digest: str
    runtime_resource_policy_digest: str
    hidden_tasks_required: bool
    minimum_prompt_count: int
    policy_version: str = "qualification.v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if type(self.reference) is not ReferenceManifest:
            raise QualificationError("qualification reference is not typed")
        for field in (
            "calibration_context_digest",
            "calibration_digest",
            "graph_requirement_digest",
            "support_policy_digest",
            "hidden_task_policy_digest",
            "runtime_resource_policy_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        metrics = tuple(_id(value, "quality metric") for value in self.required_quality_metrics)
        _ordered(metrics, metrics, "required quality metrics")
        object.__setattr__(self, "required_quality_metrics", metrics)
        from optima.eval.calibration import decimal_value
        try:
            threshold = decimal_value(self.nll_tail_threshold)
        except ValueError as exc:
            raise QualificationError(f"nll_tail_threshold: {exc}") from None
        if not 0 < threshold <= 1_000_000:
            raise QualificationError("nll_tail_threshold is outside its supported bound")
        object.__setattr__(self, "tokens_per_prompt", _integer(self.tokens_per_prompt, "tokens", 1))
        object.__setattr__(self, "topk_width", _integer(self.topk_width, "top-k width", 1))
        object.__setattr__(
            self,
            "hidden_tasks_per_prompt",
            _integer(self.hidden_tasks_per_prompt, "hidden tasks per prompt"),
        )
        object.__setattr__(
            self, "hidden_tasks_required", _boolean(self.hidden_tasks_required, "hidden tasks")
        )
        if self.hidden_tasks_required != (self.hidden_tasks_per_prompt > 0):
            raise QualificationError("hidden-task requirement and count disagree")
        object.__setattr__(
            self, "minimum_prompt_count", _integer(self.minimum_prompt_count, "prompts", 2)
        )
        if (
            self.policy_version != "qualification.v1"
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise QualificationError("unsupported qualification profile policy/schema")

    @classmethod
    def from_dict(cls, value: object) -> "QualificationProfile":
        return _load(
            cls,
            value,
            "qualification profile",
            reference=ReferenceManifest.from_dict,
            required_quality_metrics=lambda rows: tuple(_array(rows, "quality metrics")),
        )


@dataclass(frozen=True)
class SelectionCommitment(_Canonical):
    """Prompt pool and secret commitment sealed before candidate results exist."""

    _domain: ClassVar[str] = "optima.qualification.selection-commitment"
    source_plan_digest: str
    reference_manifest_digest: str
    workload_digest: str
    entropy_source_digest: str
    secret_commitment: str
    prompt_digests: tuple[str, ...]
    select_count: int
    policy_version: str = "qualification-selection.v1"
    schema_version: int = 1

    def __post_init__(self) -> None:
        for field in (
            "source_plan_digest",
            "reference_manifest_digest",
            "workload_digest",
            "entropy_source_digest",
            "secret_commitment",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        prompts = tuple(_digest(value, "prompt digest") for value in self.prompt_digests)
        _ordered(prompts, prompts, "selection prompt pool")
        object.__setattr__(self, "prompt_digests", prompts)
        count = _integer(self.select_count, "selection count", 1)
        if count > len(prompts):
            raise QualificationError("selection count exceeds its committed prompt pool")
        object.__setattr__(self, "select_count", count)
        if (
            self.policy_version != "qualification-selection.v1"
            or type(self.schema_version) is not int
            or self.schema_version != 1
        ):
            raise QualificationError("unsupported selection policy/schema")

    @classmethod
    def seal(
        cls,
        *,
        source_plan_digest: str,
        reference_manifest: ReferenceManifest,
        entropy_source_digest: str,
        prompt_digests: tuple[str, ...],
        select_count: int,
        secret: bytes,
    ) -> "SelectionCommitment":
        if type(reference_manifest) is not ReferenceManifest:
            raise QualificationError("selection reference manifest is not typed")
        if not isinstance(secret, bytes) or len(secret) < 32:
            raise QualificationError("selection commitment requires 256 secret bits")
        return cls(
            source_plan_digest,
            reference_manifest.digest,
            reference_manifest.workload_digest,
            entropy_source_digest,
            hashlib.sha256(b"optima-selection-secret-v1\0" + secret).hexdigest(),
            prompt_digests,
            select_count,
        )

    @classmethod
    def from_dict(cls, value: object) -> "SelectionCommitment":
        return _load(
            cls,
            value,
            "selection commitment",
            prompt_digests=lambda rows: tuple(_array(rows, "prompt pool")),
        )


@dataclass(frozen=True)
class SelectionEntropyReceipt(_Canonical):
    """Retained entropy value and authority receipt for a committed source."""

    _domain: ClassVar[str] = "optima.qualification.selection-entropy"
    source_digest: str
    commitment_digest: str
    entropy_digest: str
    authority_receipt_digest: str

    def __post_init__(self) -> None:
        for item in fields(self):
            object.__setattr__(self, item.name, _digest(getattr(self, item.name), item.name))

    @classmethod
    def from_dict(cls, value: object) -> "SelectionEntropyReceipt":
        return _load(cls, value, "selection entropy receipt")


@dataclass(frozen=True)
class SelectionReceipt(_Canonical):
    """Reproducible reveal using retained post-commit entropy."""

    _domain: ClassVar[str] = "optima.qualification.selection-receipt"
    commitment_digest: str
    reveal_digest: str
    entropy_receipt_digest: str
    sealed_cohort_trajectory_digest: str
    selected_prompt_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        for field in (
            "commitment_digest",
            "reveal_digest",
            "entropy_receipt_digest",
            "sealed_cohort_trajectory_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        selected = tuple(_digest(value, "selected prompt") for value in self.selected_prompt_digests)
        _ordered(selected, selected, "selected prompts")
        object.__setattr__(self, "selected_prompt_digests", selected)

    @classmethod
    def reveal(
        cls,
        commitment: SelectionCommitment,
        *,
        secret: bytes,
        entropy: SelectionEntropyReceipt,
        sealed_cohort_trajectory_digest: str,
    ) -> "SelectionReceipt":
        if type(commitment) is not SelectionCommitment or not isinstance(secret, bytes):
            raise QualificationError("selection reveal is not typed")
        reveal = hashlib.sha256(b"optima-selection-secret-v1\0" + secret).hexdigest()
        if reveal != commitment.secret_commitment:
            raise QualificationError("selection reveal does not open its commitment")
        if (
            type(entropy) is not SelectionEntropyReceipt
            or entropy.source_digest != commitment.entropy_source_digest
            or entropy.commitment_digest != commitment.digest
        ):
            raise QualificationError("selection entropy does not bind its committed source")
        trajectory = _digest(sealed_cohort_trajectory_digest, "cohort trajectory")
        ranked = sorted(
            commitment.prompt_digests,
            key=lambda prompt: canonical_digest(
                "optima.qualification.selection-key",
                {
                    "commitment_digest": commitment.digest,
                    "post_commit_entropy_digest": entropy.entropy_digest,
                    "prompt_digest": prompt,
                    "secret": reveal,
                },
            ),
        )[: commitment.select_count]
        return cls(commitment.digest, reveal, entropy.digest, trajectory, tuple(sorted(ranked)))

    def reopen(
        self, commitment: SelectionCommitment, entropy: SelectionEntropyReceipt
    ) -> "SelectionReceipt":
        if type(commitment) is not SelectionCommitment or (
            self.commitment_digest != commitment.digest
            or self.reveal_digest != commitment.secret_commitment
        ):
            raise QualificationError("selection receipt names another commitment")
        if (
            type(entropy) is not SelectionEntropyReceipt
            or entropy.digest != self.entropy_receipt_digest
            or entropy.source_digest != commitment.entropy_source_digest
            or entropy.commitment_digest != commitment.digest
        ):
            raise QualificationError("selection receipt has unbound entropy")
        ranked = sorted(
            commitment.prompt_digests,
            key=lambda prompt: canonical_digest(
                "optima.qualification.selection-key",
                {
                    "commitment_digest": commitment.digest,
                    "post_commit_entropy_digest": entropy.entropy_digest,
                    "prompt_digest": prompt,
                    "secret": self.reveal_digest,
                },
            ),
        )[: commitment.select_count]
        if self.selected_prompt_digests != tuple(sorted(ranked)):
            raise QualificationError("selection receipt does not reproduce")
        return self

    @classmethod
    def from_dict(cls, value: object) -> "SelectionReceipt":
        return _load(
            cls,
            value,
            "selection receipt",
            selected_prompt_digests=lambda rows: tuple(_array(rows, "selected prompts")),
        )


def _validated_topk_position(position: object) -> list[list[object]]:
    """Validate one retained distribution while accepting legitimate ties.

    Quantized inference can emit bit-identical log-probabilities for more than
    one token. A tie is valid evidence. Preserve runtime order because its first
    entry is the rollout's observed top-one identity.
    """

    if not isinstance(position, (tuple, list)) or not position:
        raise QualificationError("trajectory top-k position is malformed")
    entries: list[tuple[float, int]] = []
    for entry in position:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise QualificationError("trajectory top-k entry is malformed")
        logprob, token_id = entry
        if (
            isinstance(logprob, bool)
            or not isinstance(logprob, (int, float))
            or not math.isfinite(float(logprob))
            or type(token_id) is not int
            or token_id < 0
        ):
            raise QualificationError("trajectory top-k entry is invalid")
        entries.append((float(logprob), token_id))
    if len({token_id for _logprob, token_id in entries}) != len(entries):
        raise QualificationError("trajectory top-k contains duplicate tokens")
    if any(left[0] < right[0] for left, right in zip(entries, entries[1:])):
        raise QualificationError("trajectory top-k order is invalid")
    return [[format(logprob, ".17g"), token_id] for logprob, token_id in entries]


def _trajectory_rows(lifecycle: object):
    from optima.eval.marginal_runtime import MarginalLifecycleEvidence
    from optima.eval.oci_session_protocol import PromptEvidence
    from optima.eval.scoring import marginal_workload_digest

    if type(lifecycle) is not MarginalLifecycleEvidence:
        raise QualificationError("trajectory lifecycle is not typed")
    plan = lifecycle.prepared.baseline_session_plan
    executions = (lifecycle.baseline_before,) + tuple(
        row.execution for row in lifecycle.candidates
    ) + (lifecycle.baseline_after,)
    workload = marginal_workload_digest(plan)
    rows = []
    for batch_index, prompts in enumerate(plan.prompt_batches):
        for prompt_index, prompt in enumerate(prompts):
            occurrence = canonical_digest(
                "optima.qualification.prompt-occurrence",
                {
                    "batch_index": batch_index,
                    "prompt_index": prompt_index,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "workload_digest": workload,
                },
            )
            frames = []
            for execution in executions:
                evidence = execution.session.batches[batch_index].evidence.prompts[prompt_index]
                if (
                    type(evidence) is not PromptEvidence
                    or len(evidence.output_ids) != plan.max_new_tokens
                    or len(evidence.top_logprobs) != plan.max_new_tokens
                    or any(type(token) is not int or token < 0 for token in evidence.output_ids)
                    or any(len(position) != plan.top_logprobs_num for position in evidence.top_logprobs)
                ):
                    raise QualificationError("trajectory token/top-k coverage differs from workload")
                topk = []
                for position in evidence.top_logprobs:
                    topk.append(_validated_topk_position(position))
                frames.append({"output_ids": list(evidence.output_ids), "top_logprobs": topk})
            rows.append((occurrence, frames))
    return workload, tuple(rows)


def lifecycle_prompt_digests(lifecycle: object) -> tuple[str, ...]:
    """Canonical prompt-occurrence pool fixed before any quality selection."""

    return tuple(sorted(row[0] for row in _trajectory_rows(lifecycle)[1]))


def cohort_trajectory_digest(lifecycle: object) -> str:
    """Bind every B/C/B-prime retained token and top-k frame in execution order."""

    workload, rows = _trajectory_rows(lifecycle)
    return canonical_digest(
        "optima.qualification.cohort-trajectories",
        {"workload_digest": workload, "prompts": [[key, frames] for key, frames in rows]},
    )


def candidate_lifecycle_digest(
    lifecycle: object, *, selected_delta_digest: str
) -> str:
    """Bind the exact retained B/C/B-prime execution used by one qualifier."""

    from optima.eval.marginal_runtime import MarginalLifecycleEvidence

    if type(lifecycle) is not MarginalLifecycleEvidence:
        raise QualificationError("candidate lifecycle is not typed")
    candidates = tuple(
        row
        for row in lifecycle.candidates
        if row.arm.selected_delta_digest == selected_delta_digest
    )
    if len(candidates) != 1:
        raise QualificationError("candidate lifecycle is absent or ambiguous")
    candidate = candidates[0]
    executions = (
        lifecycle.baseline_before,
        candidate.execution,
        lifecycle.baseline_after,
    )

    def execution_row(execution):
        session = execution.session
        return {
            "device": [
                [row.launch_id, row.sequence] for row in execution.device_receipts
            ],
            "launch_digest": execution.launch_digest,
            "native_publication_digest": execution.native_publication_digest,
            "requests": [
                [
                    row.request_id,
                    row.nonce,
                    row.token_numerator,
                    format(row.request_started_at, ".17g"),
                    format(row.response_completed_at, ".17g"),
                ]
                for row in session.batches
            ],
            "resource_policy_digest": execution.resource_policy_digest,
            "runtime_argv_sha256": execution.runtime_argv_sha256,
            "session_id": session.session_id,
        }

    return canonical_digest(
        "optima.qualification.candidate-lifecycle",
        {
            "arm_digest": candidate.arm.digest,
            "cohort_trajectory_digest": cohort_trajectory_digest(lifecycle),
            "executions": [execution_row(row) for row in executions],
            "selected_delta_digest": _digest(
                selected_delta_digest, "selected delta"
            ),
            "source_digest": lifecycle.source.digest,
        },
    )


def qualification_identity_digest(
    profile: QualificationProfile,
    *,
    graph_requirement: GraphVerificationRequirement,
    selection: SelectionReceipt,
    calibration: object,
    candidate_lifecycle: str,
    t_session: object,
    t_request_sha256: str,
    selected_delta_digest: str,
) -> str:
    """Derive the raw-quality identity from typed validator authority."""

    from optima.eval.calibration import CalibrationManifest
    from optima.eval.oci_reference_session import ReferenceSessionEvidence

    if (
        type(profile) is not QualificationProfile
        or type(graph_requirement) is not GraphVerificationRequirement
        or type(selection) is not SelectionReceipt
        or type(calibration) is not CalibrationManifest
        or type(t_session) is not ReferenceSessionEvidence
    ):
        raise QualificationError("qualification identity inputs are not typed")
    return canonical_digest(
        "optima.qualification.candidate-identity",
        {
            "calibration_digest": calibration.digest,
            "candidate_lifecycle_digest": _digest(
                candidate_lifecycle, "candidate lifecycle"
            ),
            "graph_requirement_digest": graph_requirement.digest,
            "profile_digest": profile.digest,
            "selected_delta_digest": _digest(
                selected_delta_digest, "selected delta"
            ),
            "selection_digest": selection.digest,
            "t_session_digest": t_session.digest,
            "t_request_sha256": _digest(t_request_sha256, "T request SHA-256"),
        },
    )


def selected_trajectory_digest(
    lifecycle: object,
    *,
    selected_delta_digest: str,
    selected_prompt_digests: tuple[str, ...],
) -> str:
    """Bind selected B/C/B-prime frames for one exact candidate arm."""

    candidates = tuple(row.arm.selected_delta_digest for row in lifecycle.candidates)
    if candidates.count(selected_delta_digest) != 1:
        raise QualificationError("selected trajectory candidate is absent or ambiguous")
    candidate_index = candidates.index(selected_delta_digest) + 1
    workload, rows = _trajectory_rows(lifecycle)
    selected = tuple(selected_prompt_digests)
    if selected != tuple(sorted(set(selected))) or not set(selected) <= {row[0] for row in rows}:
        raise QualificationError("selected trajectory prompts differ from the lifecycle")
    by_prompt = dict(rows)
    return canonical_digest(
        "optima.qualification.selected-trajectories",
        {
            "selected_delta_digest": _digest(selected_delta_digest, "selected delta"),
            "workload_digest": workload,
            "prompts": [
                [prompt, [by_prompt[prompt][index] for index in (0, candidate_index, -1)]]
                for prompt in selected
            ],
        },
    )


def selected_trajectory_projection_digest(
    lifecycle: object,
    *,
    selected_delta_digest: str,
    selected_prompt_digests: tuple[str, ...],
) -> str:
    """Bind raw-quality-checkable token, support, and true-argmax facts."""

    candidates = tuple(row.arm.selected_delta_digest for row in lifecycle.candidates)
    if candidates.count(selected_delta_digest) != 1:
        raise QualificationError("selected trajectory candidate is absent or ambiguous")
    candidate_index = candidates.index(selected_delta_digest) + 1
    _, rows = _trajectory_rows(lifecycle)
    selected = tuple(selected_prompt_digests)
    if selected != tuple(sorted(set(selected))) or not set(selected) <= {row[0] for row in rows}:
        raise QualificationError("selected trajectory prompts differ from the lifecycle")
    by_prompt = dict(rows)

    from optima.eval.reference_quality import (
        distribution_from_f32_logprobs,
        retained_support_policy_digest,
    )

    def project(frame):
        distributions = []
        for position in frame["top_logprobs"]:
            by_token = sorted(position, key=lambda row: row[1])
            distributions.append(
                distribution_from_f32_logprobs(
                    tuple(row[1] for row in by_token),
                    tuple(float(row[0]) for row in by_token),
                    true_argmax_token_id=position[0][1],
                ).to_dict()
            )
        return {
            "output_ids": frame["output_ids"],
            "rollout_topk": distributions,
        }

    return canonical_digest(
        "optima.qualification.selected-trajectory-projection",
        {
            "support_policy_digest": retained_support_policy_digest(),
            "prompts": [
                {
                    "prompt": prompt,
                    "rollouts": [
                        project(by_prompt[prompt][index])
                        for index in (0, candidate_index, -1)
                    ],
                }
                for prompt in selected
            ],
        },
    )


def derived_hidden_task_plan_digest(
    profile: QualificationProfile, selected_prompt_digests: tuple[str, ...]
) -> str:
    """Derive opaque task identities from validator-owned corpus/judge policy."""

    if type(profile) is not QualificationProfile:
        raise QualificationError("hidden-task profile is not typed")
    prompts = tuple(selected_prompt_digests)
    if prompts != tuple(sorted(set(prompts))):
        raise QualificationError("hidden-task prompts are not canonical")
    rows = []
    for prompt in prompts:
        tasks = sorted(canonical_digest("optima.qualification.hidden-task", {
            "corpus": profile.reference.hidden_corpus_commitment,
            "judge": profile.reference.hidden_judge_digest,
            "policy": profile.hidden_task_policy_digest,
            "prompt": _digest(prompt, "hidden-task prompt"),
            "index": index,
        }) for index in range(profile.hidden_tasks_per_prompt))
        rows.append({"prompt": prompt, "tasks": tasks})
    return canonical_digest("optima.qualification.hidden-task-plan", rows)


def _selected_prompt_texts(lifecycle: object) -> dict[str, str]:
    from optima.eval.scoring import marginal_workload_digest

    plan = lifecycle.prepared.baseline_session_plan
    workload = marginal_workload_digest(plan)
    result = {}
    for batch_index, prompts in enumerate(plan.prompt_batches):
        for prompt_index, prompt in enumerate(prompts):
            occurrence = canonical_digest(
                "optima.qualification.prompt-occurrence",
                {
                    "batch_index": batch_index,
                    "prompt_index": prompt_index,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "workload_digest": workload,
                },
            )
            result[occurrence] = prompt
    return result


def _validate_teacher_source(
    raw: object,
    execution: object,
    lifecycle: object,
    *,
    reference_request_sha256: str,
) -> None:
    from optima.eval.oci_backend import PristineReferenceExecutionEvidence
    from optima.eval.reference_quality import (
        ReferenceQualityRawArtifact,
        distribution_from_f32_logprobs,
        target_nll_from_f32,
    )

    if (
        type(raw) is not ReferenceQualityRawArtifact
        or type(execution) is not PristineReferenceExecutionEvidence
        or execution.schema != "optima.oci-pristine-reference-execution.v1"
    ):
        raise QualificationError("raw quality or pristine execution is not authoritative")
    request_digest = _digest(reference_request_sha256, "T request SHA-256")
    exchanges = tuple(
        row for row in execution.session.exchanges
        if row.request_sha256 == request_digest
    )
    if len(exchanges) != 1:
        raise QualificationError("pristine T request is absent or ambiguous")
    exchange = exchanges[0]
    by_prompt = {}
    for request, evidence in zip(
        exchange.request.prompts, exchange.evidence.prompts, strict=True
    ):
        if request.prompt_digest in by_prompt:
            raise QualificationError("pristine T repeated a selected prompt")
        by_prompt[request.prompt_digest] = (request, evidence)
    expected_text = _selected_prompt_texts(lifecycle)
    if set(by_prompt) != set(raw.binding.selected_prompt_digests):
        raise QualificationError("pristine T prompt coverage differs from selection")
    for prompt in raw.prompts:
        request, evidence = by_prompt[prompt.prompt_digest]
        if request.prompt != expected_text.get(prompt.prompt_digest):
            raise QualificationError("pristine T prompt text differs from lifecycle")
        for rollout, role_input, role_evidence in zip(
            (prompt.baseline, prompt.candidate, prompt.stock_control),
            request.roles,
            evidence.roles,
            strict=True,
        ):
            for token, output_id, support, teacher in zip(
                rollout.tokens,
                role_input.output_ids,
                role_input.supports,
                role_evidence.tokens,
                strict=True,
            ):
                expected = distribution_from_f32_logprobs(
                    support,
                    teacher.support_logprobs,
                    true_argmax_token_id=teacher.true_argmax_token_id,
                )
                if (
                    token.token_id != output_id
                    or token.target_nll != target_nll_from_f32(teacher.target_logprob)
                    or token.teacher_topk != expected
                ):
                    raise QualificationError("raw teacher evidence differs from pristine T")


def validate_quality_binding(
    profile: QualificationProfile,
    raw_artifact: object,
    lifecycle: object,
    *,
    selected_delta_digest: str,
    commitment: SelectionCommitment,
    entropy: SelectionEntropyReceipt,
    selection: SelectionReceipt,
    calibration: object,
    graph_requirement: GraphVerificationRequirement,
    reference_execution: object,
    reference_request_sha256: str,
):
    """Project frozen workload/trajectory coverage onto one raw T binding."""

    from optima.eval.calibration import CalibrationContext, CalibrationManifest
    from optima.eval.oci_backend import PristineReferenceExecutionEvidence
    from optima.eval.reference_quality import (
        ReferenceQualityRawArtifact,
        retained_support_policy_digest,
    )

    if (
        type(profile) is not QualificationProfile
        or type(raw_artifact) is not ReferenceQualityRawArtifact
        or type(calibration) is not CalibrationManifest
        or type(graph_requirement) is not GraphVerificationRequirement
        or type(reference_execution) is not PristineReferenceExecutionEvidence
    ):
        raise QualificationError("quality profile/binding is not typed")
    binding = raw_artifact.binding
    t_session = reference_execution.session
    selection.reopen(commitment, entropy)
    plan = lifecycle.prepared.baseline_session_plan
    candidates = tuple(
        row for row in lifecycle.candidates
        if row.arm.selected_delta_digest == selected_delta_digest
    )
    if len(candidates) != 1:
        raise QualificationError("quality candidate lifecycle is absent or ambiguous")
    candidate = candidates[0]
    arm = candidate.arm
    expected_graph_binding = (
        arm.digest,
        candidate.candidate.launch.digest,
        arm.transition.replacement.digest,
        arm.selected_delta_digest,
        arm.transition.target_id,
        arm.transition.target_spec_digest,
        arm.candidate.catalog_digest,
    )
    actual_graph_binding = (
        graph_requirement.binding.marginal_arm_digest,
        graph_requirement.binding.candidate_launch_digest,
        graph_requirement.binding.contribution_ref_digest,
        graph_requirement.binding.selected_delta_digest,
        graph_requirement.binding.target_id,
        graph_requirement.binding.target_spec_digest,
        graph_requirement.binding.catalog_digest,
    )
    expected_calibration_context = CalibrationContext(
        profile.reference.digest,
        profile.reference.arena_digest,
        profile.reference.runtime_digest,
        profile.reference.base_engine_digest,
        profile.reference.model_revision_digest,
        profile.reference.model_manifest_digest,
        profile.reference.model_content_digest,
        profile.reference.logical_hardware_digest,
        profile.reference.workload_digest,
        graph_requirement.binding.verification_policy_digest,
        profile.reference.controller_distribution_digest,
    )
    lifecycle_digest = candidate_lifecycle_digest(
        lifecycle, selected_delta_digest=selected_delta_digest
    )
    identity_digest = qualification_identity_digest(
        profile,
        graph_requirement=graph_requirement,
        selection=selection,
        calibration=calibration,
        candidate_lifecycle=lifecycle_digest,
        t_session=t_session,
        t_request_sha256=reference_request_sha256,
        selected_delta_digest=selected_delta_digest,
    )
    if (
        commitment.source_plan_digest != lifecycle.source.digest
        or commitment.reference_manifest_digest != profile.reference.digest
        or commitment.workload_digest != profile.reference.workload_digest
        or commitment.entropy_source_digest != profile.reference.selection_policy_digest
        or commitment.prompt_digests != lifecycle_prompt_digests(lifecycle)
        or selection.sealed_cohort_trajectory_digest != cohort_trajectory_digest(lifecycle)
        or binding.selected_trajectory_digest != selected_trajectory_digest(
            lifecycle,
            selected_delta_digest=selected_delta_digest,
            selected_prompt_digests=selection.selected_prompt_digests,
        )
        or binding.selected_trajectory_projection_digest
        != selected_trajectory_projection_digest(
            lifecycle,
            selected_delta_digest=selected_delta_digest,
            selected_prompt_digests=selection.selected_prompt_digests,
        )
        or binding.selected_prompt_digests != selection.selected_prompt_digests
        or binding.hidden_task_plan_digest
        != derived_hidden_task_plan_digest(profile, selection.selected_prompt_digests)
        or binding.qualification_identity_digest
        != identity_digest
        or binding.candidate_lifecycle_digest
        != lifecycle_digest
        or binding.t_session_digest != t_session.digest
        or binding.t_request_sha256
        != _digest(reference_request_sha256, "T request SHA-256")
        or t_session.reference_manifest_digest != profile.reference.digest
        or t_session.launch_digest != profile.reference.pristine_launch_digest
        or binding.reference_manifest_digest != profile.reference.digest
        or binding.calibration_digest != profile.calibration_digest
        or calibration.digest != profile.calibration_digest
        or calibration.context != expected_calibration_context
        or calibration.context.digest != profile.calibration_context_digest
        or tuple(row.name for row in calibration.quality_metrics)
        != profile.required_quality_metrics
        or graph_requirement.digest != profile.graph_requirement_digest
        or actual_graph_binding != expected_graph_binding
        or binding.selection_digest != selection.digest
        or (binding.tokens_per_prompt, binding.topk_width, binding.hidden_tasks_per_prompt)
        != (profile.tokens_per_prompt, profile.topk_width, profile.hidden_tasks_per_prompt)
        or (plan.max_new_tokens, plan.top_logprobs_num)
        != (profile.tokens_per_prompt, profile.topk_width)
        or binding.support_policy_digest != profile.support_policy_digest
        or profile.support_policy_digest != retained_support_policy_digest()
        or binding.nll_tail_threshold != profile.nll_tail_threshold
        or commitment.select_count < profile.minimum_prompt_count
    ):
        raise QualificationError("quality binding differs from frozen workload/trajectories")
    _validate_teacher_source(
        raw_artifact,
        reference_execution,
        lifecycle,
        reference_request_sha256=reference_request_sha256,
    )
    return raw_artifact


__all__ = [
    "GRAPH_EVIDENCE_DOMAIN",
    "GRAPH_EVIDENCE_MEDIA_TYPE",
    "GRAPH_EVIDENCE_SCHEMA",
    "GRAPH_QUALIFICATION_POLICY_VERSION",
    "GRAPH_QUALIFICATION_SCHEMA_VERSION",
    "GraphMemberEvidence",
    "GraphShapeEvidence",
    "GraphVariantEvidence",
    "GraphVariantRequirement",
    "GraphVerificationBinding",
    "GraphVerificationEvidenceRef",
    "GraphVerificationGrade",
    "GraphVerificationMemberBinding",
    "GraphVerificationRawEvidence",
    "GraphVerificationRequirement",
    "QualificationProfile",
    "QualificationDecision",
    "QualificationError",
    "ReferenceManifest",
    "SelectionCommitment",
    "SelectionEntropyReceipt",
    "SelectionReceipt",
    "candidate_lifecycle_digest",
    "cohort_trajectory_digest",
    "derived_hidden_task_plan_digest",
    "lifecycle_prompt_digests",
    "qualification_identity_digest",
    "reopen_graph_verification",
    "regrade_graph_verification",
    "selected_trajectory_digest",
    "selected_trajectory_projection_digest",
    "validate_quality_binding",
]
