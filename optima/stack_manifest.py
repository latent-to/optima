"""Immutable identities for evaluation incumbents and reviewed releases.

The types in this module describe content; they do not resolve artifacts,
materialize files, launch engines, or establish attribution authenticity.
Structural parsing is context-free.  A validator must separately call
``validate_against`` with its explicit expected runtime/catalog context.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeAlias

from optima.eval.evidence_store import EvidenceArtifactRef
from optima.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
)


CONTRIBUTION_REF_SCHEMA_VERSION = 1
INTEGRATION_REVIEW_SCHEMA_VERSION = 1
STACK_MANIFEST_SCHEMA_VERSION = 1
EVALUATION_STACK_POLICY_VERSION = "evaluation-stack.v1"
ENGINE_RELEASE_POLICY_VERSION = "engine-release.v1"
# Compatibility name for the evaluation planner, whose arms are evaluation-only.
STACK_POLICY_VERSION = EVALUATION_STACK_POLICY_VERSION

_TARGET_RE = re.compile(r"^[0-9A-Za-z._-]+$")


class StackManifestError(ValueError):
    """A contribution or stack manifest is malformed or stale."""


def _target_id(value: object, *, field: str = "target_id") -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or not _TARGET_RE.fullmatch(value)
    ):
        raise StackManifestError(f"{field} must be a canonical target ID")
    return value


def _digest(value: object, *, field: str) -> str:
    try:
        return require_sha256_hex(value, field=field)
    except StackIdentityError as exc:
        raise StackManifestError(str(exc)) from exc


def _current_version(value: object, *, field: str, expected: object) -> object:
    if value != expected or type(value) is not type(expected):
        raise StackManifestError(f"unsupported {field} {value!r}")
    return value


def _strict_object(
    value: object, *, fields: frozenset[str], name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise StackManifestError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise StackManifestError(f"{name} keys must be strings")
    actual = frozenset(value)
    if actual != fields:
        missing = tuple(sorted(fields - actual))
        extra = tuple(sorted(actual - fields))
        raise StackManifestError(
            f"{name} fields mismatch: missing={missing!r}, extra={extra!r}"
        )
    return value


def _catalog_json(snapshot: object) -> bytes:
    if not isinstance(snapshot, Mapping) or not snapshot:
        raise StackManifestError("catalog_snapshot must be a non-empty object")
    try:
        return canonical_json_bytes(snapshot)
    except StackIdentityError as exc:
        raise StackManifestError(f"invalid catalog_snapshot: {exc}") from exc


def _catalog_binding(snapshot: object, digest: str) -> tuple[bytes, str]:
    canonical = _catalog_json(snapshot)
    supplied = _digest(digest, field="catalog_digest")
    decoded = json.loads(canonical)
    computed = canonical_digest("optima.target-catalog", decoded)
    if supplied != computed:
        raise StackManifestError(
            "catalog_digest does not match the embedded catalog_snapshot"
        )
    return canonical, supplied


def _spec_rows(value: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise StackManifestError("target_spec_digests must be an object")
    rows: list[tuple[str, str]] = []
    for raw_target, raw_digest in value.items():
        target = _target_id(raw_target, field="target_spec_digests key")
        rows.append(
            (target, _digest(raw_digest, field=f"target spec {target!r} digest"))
        )
    if len({target for target, _ in rows}) != len(rows):
        raise StackManifestError("target_spec_digests contains duplicate targets")
    return tuple(sorted(rows))


def _catalog_spec_rows(catalog_json: bytes) -> tuple[tuple[str, str], ...]:
    catalog = json.loads(catalog_json)
    targets = catalog.get("targets")
    if not isinstance(targets, list) or not targets:
        raise StackManifestError("catalog_snapshot targets must be a non-empty list")
    rows: list[tuple[str, str]] = []
    for index, row in enumerate(targets):
        if not isinstance(row, Mapping):
            raise StackManifestError(f"catalog target {index} must be an object")
        target = _target_id(row.get("target_id"), field=f"catalog target {index} ID")
        rows.append((target, canonical_digest("optima.target-spec", row)))
    if len({target for target, _ in rows}) != len(rows):
        raise StackManifestError("catalog_snapshot contains duplicate target IDs")
    return tuple(sorted(rows))


def _context_spec_rows(catalog_json: bytes, supplied: object) -> tuple[tuple[str, str], ...]:
    rows = _spec_rows(supplied)
    if rows != _catalog_spec_rows(catalog_json):
        raise StackManifestError(
            "target_spec_digests do not match the complete catalog_snapshot"
        )
    return rows


def _selected_delta(target_id: str, target_spec: str, payload: str) -> str:
    return canonical_digest(
        "optima.contribution.selected_delta",
        {
            "selected_payload_digest": payload,
            "target_id": target_id,
            "target_spec_digest": target_spec,
        },
    )


@dataclass(frozen=True)
class ProposalContributionRef:
    """One hostile proposal, with artifact, selected delta, and attribution separate."""

    target_id: str
    target_spec_digest: str
    artifact_digest: str
    selected_payload_digest: str
    attribution_digest: str
    schema_version: int = CONTRIBUTION_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _target_id(self.target_id))
        for field in (
            "target_spec_digest",
            "artifact_digest",
            "selected_payload_digest",
            "attribution_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        _current_version(
            self.schema_version,
            field="proposal contribution schema_version",
            expected=CONTRIBUTION_REF_SCHEMA_VERSION,
        )

    @property
    def selected_delta_digest(self) -> str:
        return _selected_delta(
            self.target_id, self.target_spec_digest, self.selected_payload_digest
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.contribution.proposal", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact_digest": self.artifact_digest,
            "attribution_digest": self.attribution_digest,
            "schema_version": self.schema_version,
            "selected_payload_digest": self.selected_payload_digest,
            "target_id": self.target_id,
            "target_spec_digest": self.target_spec_digest,
            "type": "proposal",
        }

    @classmethod
    def from_dict(cls, value: object) -> "ProposalContributionRef":
        row = _strict_object(
            value,
            fields=frozenset(
                {
                    "artifact_digest",
                    "attribution_digest",
                    "schema_version",
                    "selected_payload_digest",
                    "target_id",
                    "target_spec_digest",
                    "type",
                }
            ),
            name="proposal contribution",
        )
        if row["type"] != "proposal":
            raise StackManifestError("proposal contribution type must be 'proposal'")
        return cls(
            target_id=row["target_id"],  # type: ignore[arg-type]
            target_spec_digest=row["target_spec_digest"],  # type: ignore[arg-type]
            artifact_digest=row["artifact_digest"],  # type: ignore[arg-type]
            selected_payload_digest=row["selected_payload_digest"],  # type: ignore[arg-type]
            attribution_digest=row["attribution_digest"],  # type: ignore[arg-type]
            schema_version=row["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class IntegrationReviewArtifacts:
    """Retained evidence references used by one source-integration review."""

    primary_attempt_ref: EvidenceArtifactRef
    reproduction_attempt_ref: EvidenceArtifactRef
    license_evidence_ref: EvidenceArtifactRef
    provenance_evidence_ref: EvidenceArtifactRef
    security_review_ref: EvidenceArtifactRef
    compatibility_evidence_ref: EvidenceArtifactRef
    test_evidence_ref: EvidenceArtifactRef

    def __post_init__(self) -> None:
        for field in self.__dataclass_fields__:
            if type(getattr(self, field)) is not EvidenceArtifactRef:
                raise StackManifestError(
                    f"integration review artifact {field} is not exactly typed"
                )
        if self.primary_attempt_ref == self.reproduction_attempt_ref:
            raise StackManifestError(
                "integration review artifacts reuse the primary attempt"
            )
        expected = {
            "primary_attempt_ref": (
                "qualification.cohort-attempt",
                "optima.qualification.cohort-attempt.v1",
            ),
            "reproduction_attempt_ref": (
                "qualification.cohort-attempt",
                "optima.qualification.cohort-attempt.v1",
            ),
            "license_evidence_ref": (
                "integration.license",
                "optima.integration.license.v1",
            ),
            "provenance_evidence_ref": (
                "integration.provenance",
                "optima.integration.provenance.v1",
            ),
            "security_review_ref": (
                "integration.security-review",
                "optima.integration.security-review.v1",
            ),
            "compatibility_evidence_ref": (
                "integration.compatibility",
                "optima.integration.compatibility.v1",
            ),
            "test_evidence_ref": (
                "integration.tests",
                "optima.integration.tests.v1",
            ),
        }
        for field, (domain, schema) in expected.items():
            reference = getattr(self, field)
            if (
                reference.domain != domain
                or reference.schema != schema
                or reference.media_type != "application/json"
            ):
                raise StackManifestError(
                    f"integration review artifact {field} has the wrong domain/schema"
                )
        references = tuple(getattr(self, field) for field in self.__dataclass_fields__)
        if len({reference.sha256 for reference in references}) != len(references):
            raise StackManifestError(
                "integration review artifact digests must be pairwise distinct"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            field: getattr(self, field).to_dict()
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> "IntegrationReviewArtifacts":
        row = _strict_object(
            value,
            fields=frozenset(cls.__dataclass_fields__),
            name="integration review artifacts",
        )
        return cls(
            **{
                field: EvidenceArtifactRef.from_dict(row[field])
                for field in cls.__dataclass_fields__
            }
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.integration-review.artifacts", self.to_dict())


@dataclass(frozen=True)
class IntegrationReviewRecord:
    """Reviewed promotion from one reproduced crown to ordinary Optima source.

    The record is deliberately chain-independent after construction: chain/crown
    identities remain immutable provenance, while approval, source, licenses,
    compatibility, security, and tests are owned by source control and release review.
    """

    target_id: str
    target_spec_digest: str
    proposal_contribution_digest: str
    settlement_candidate_digest: str
    settlement_evidence_digest: str
    crown_event_digest: str
    primary_attempt_digest: str
    reproduction_attempt_digest: str
    integrated_source_tree_digest: str
    selected_payload_digest: str
    attribution_digest: str
    license_evidence_digest: str
    provenance_evidence_digest: str
    security_review_digest: str
    compatibility_evidence_digest: str
    test_evidence_digest: str
    artifacts: IntegrationReviewArtifacts
    reviewer: str
    review_commit: str
    approved: bool = True
    schema_version: int = INTEGRATION_REVIEW_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _target_id(self.target_id))
        for field in (
            "target_spec_digest",
            "proposal_contribution_digest",
            "settlement_candidate_digest",
            "settlement_evidence_digest",
            "crown_event_digest",
            "primary_attempt_digest",
            "reproduction_attempt_digest",
            "integrated_source_tree_digest",
            "selected_payload_digest",
            "attribution_digest",
            "license_evidence_digest",
            "provenance_evidence_digest",
            "security_review_digest",
            "compatibility_evidence_digest",
            "test_evidence_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        if self.primary_attempt_digest == self.reproduction_attempt_digest:
            raise StackManifestError("integration review requires independent attempts")
        if type(self.artifacts) is not IntegrationReviewArtifacts:
            raise StackManifestError("integration review lacks retained artifact references")
        expected_artifacts = (
            ("primary_attempt_digest", self.artifacts.primary_attempt_ref),
            ("reproduction_attempt_digest", self.artifacts.reproduction_attempt_ref),
            ("license_evidence_digest", self.artifacts.license_evidence_ref),
            ("provenance_evidence_digest", self.artifacts.provenance_evidence_ref),
            ("security_review_digest", self.artifacts.security_review_ref),
            ("compatibility_evidence_digest", self.artifacts.compatibility_evidence_ref),
            ("test_evidence_digest", self.artifacts.test_evidence_ref),
        )
        if any(getattr(self, field) != reference.sha256 for field, reference in expected_artifacts):
            raise StackManifestError(
                "integration review digest differs from retained artifact references"
            )
        if (
            not isinstance(self.reviewer, str)
            or not self.reviewer
            or self.reviewer.strip() != self.reviewer
            or len(self.reviewer) > 256
            or any(char in self.reviewer for char in "\x00\r\n")
        ):
            raise StackManifestError("integration reviewer identity is malformed")
        if not isinstance(self.review_commit, str) or re.fullmatch(
            r"[0-9a-f]{40}", self.review_commit
        ) is None:
            raise StackManifestError("integration review_commit must be a full Git SHA-1")
        if self.approved is not True:
            raise StackManifestError("only approved integration records may enter a release")
        _current_version(
            self.schema_version,
            field="integration review schema_version",
            expected=INTEGRATION_REVIEW_SCHEMA_VERSION,
        )

    @property
    def selected_delta_digest(self) -> str:
        return _selected_delta(
            self.target_id, self.target_spec_digest, self.selected_payload_digest
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.contribution.integration-review", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            field: (
                self.artifacts.to_dict()
                if field == "artifacts"
                else getattr(self, field)
            )
            for field in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: object) -> "IntegrationReviewRecord":
        row = _strict_object(
            value,
            fields=frozenset(cls.__dataclass_fields__),
            name="integration review",
        )
        values = dict(row)
        values["artifacts"] = IntegrationReviewArtifacts.from_dict(row["artifacts"])
        return cls(**values)  # type: ignore[arg-type]

    def integrated_ref(self) -> "IntegratedContributionRef":
        return IntegratedContributionRef(
            target_id=self.target_id,
            target_spec_digest=self.target_spec_digest,
            integrated_source_tree_digest=self.integrated_source_tree_digest,
            selected_payload_digest=self.selected_payload_digest,
            attribution_digest=self.attribution_digest,
            integration_record_digest=self.digest,
        )

    def require_ref(self, value: "IntegratedContributionRef") -> None:
        if not isinstance(value, IntegratedContributionRef) or value != self.integrated_ref():
            raise StackManifestError("integrated contribution differs from its review record")


@dataclass(frozen=True)
class IntegratedContributionRef:
    """One reviewed source contribution authorized by an integration record."""

    target_id: str
    target_spec_digest: str
    integrated_source_tree_digest: str
    selected_payload_digest: str
    attribution_digest: str
    integration_record_digest: str
    schema_version: int = CONTRIBUTION_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _target_id(self.target_id))
        for field in (
            "target_spec_digest",
            "integrated_source_tree_digest",
            "selected_payload_digest",
            "attribution_digest",
            "integration_record_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        _current_version(
            self.schema_version,
            field="integrated contribution schema_version",
            expected=CONTRIBUTION_REF_SCHEMA_VERSION,
        )

    @property
    def selected_delta_digest(self) -> str:
        return _selected_delta(
            self.target_id, self.target_spec_digest, self.selected_payload_digest
        )

    @property
    def digest(self) -> str:
        return canonical_digest("optima.contribution.integrated", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "attribution_digest": self.attribution_digest,
            "integrated_source_tree_digest": self.integrated_source_tree_digest,
            "integration_record_digest": self.integration_record_digest,
            "schema_version": self.schema_version,
            "selected_payload_digest": self.selected_payload_digest,
            "target_id": self.target_id,
            "target_spec_digest": self.target_spec_digest,
            "type": "integrated",
        }

    @classmethod
    def from_dict(cls, value: object) -> "IntegratedContributionRef":
        row = _strict_object(
            value,
            fields=frozenset(
                {
                    "attribution_digest",
                    "integrated_source_tree_digest",
                    "integration_record_digest",
                    "schema_version",
                    "selected_payload_digest",
                    "target_id",
                    "target_spec_digest",
                    "type",
                }
            ),
            name="integrated contribution",
        )
        if row["type"] != "integrated":
            raise StackManifestError(
                "integrated contribution type must be 'integrated'"
            )
        return cls(
            target_id=row["target_id"],  # type: ignore[arg-type]
            target_spec_digest=row["target_spec_digest"],  # type: ignore[arg-type]
            integrated_source_tree_digest=row[
                "integrated_source_tree_digest"
            ],  # type: ignore[arg-type]
            selected_payload_digest=row["selected_payload_digest"],  # type: ignore[arg-type]
            attribution_digest=row["attribution_digest"],  # type: ignore[arg-type]
            integration_record_digest=row["integration_record_digest"],  # type: ignore[arg-type]
            schema_version=row["schema_version"],  # type: ignore[arg-type]
        )


ContributionRef: TypeAlias = ProposalContributionRef | IntegratedContributionRef


def contribution_ref_from_dict(value: object) -> ContributionRef:
    if not isinstance(value, Mapping):
        raise StackManifestError("contribution must be an object")
    kind = value.get("type")
    if kind == "proposal":
        return ProposalContributionRef.from_dict(value)
    if kind == "integrated":
        return IntegratedContributionRef.from_dict(value)
    raise StackManifestError("contribution requires type 'proposal' or 'integrated'")


def _entry_rows(
    entries: Mapping[str, ContributionRef]
    | Iterable[tuple[str, ContributionRef]],
    *,
    integrated_only: bool,
) -> tuple[tuple[str, ContributionRef], ...]:
    if isinstance(entries, Mapping):
        raw_rows = tuple(entries.items())
    elif isinstance(entries, (str, bytes)):
        raise StackManifestError("entries must be a mapping or pair iterable")
    else:
        raw_rows = tuple(entries)
    rows: list[tuple[str, ContributionRef]] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise StackManifestError(f"entry {index} must be a target/ref pair")
        raw_target, ref = raw
        target = _target_id(raw_target, field=f"entry {index} target")
        if not isinstance(ref, (ProposalContributionRef, IntegratedContributionRef)):
            raise StackManifestError(f"entry {target!r} has invalid contribution type")
        if integrated_only and not isinstance(ref, IntegratedContributionRef):
            raise StackManifestError("release manifests accept integrated contributions only")
        if ref.target_id != target:
            raise StackManifestError(
                f"entry key {target!r} does not match ref target {ref.target_id!r}"
            )
        rows.append((target, ref))
    if len({target for target, _ in rows}) != len(rows):
        raise StackManifestError("entries contain duplicate targets")
    return tuple(sorted(rows, key=lambda row: row[0]))


@dataclass(frozen=True, init=False)
class EvaluationStackContext:
    """Validator-owned identities expected for one evaluation arena."""

    runtime_digest: str
    base_engine_digest: str
    arena_digest: str
    catalog_digest: str
    _catalog_json: bytes
    _target_specs: tuple[tuple[str, str], ...]

    def __init__(
        self,
        *,
        runtime_digest: str,
        base_engine_digest: str,
        arena_digest: str,
        catalog_snapshot: Mapping[str, object],
        catalog_digest: str,
        target_spec_digests: Mapping[str, str],
    ) -> None:
        object.__setattr__(
            self, "runtime_digest", _digest(runtime_digest, field="runtime_digest")
        )
        object.__setattr__(
            self,
            "base_engine_digest",
            _digest(base_engine_digest, field="base_engine_digest"),
        )
        object.__setattr__(
            self, "arena_digest", _digest(arena_digest, field="arena_digest")
        )
        catalog_json, checked_catalog_digest = _catalog_binding(
            catalog_snapshot, catalog_digest
        )
        object.__setattr__(self, "catalog_digest", checked_catalog_digest)
        object.__setattr__(self, "_catalog_json", catalog_json)
        object.__setattr__(
            self, "_target_specs", _context_spec_rows(catalog_json, target_spec_digests)
        )

    @property
    def catalog_snapshot(self) -> dict[str, Any]:
        return json.loads(self._catalog_json)

    @property
    def target_spec_digests(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._target_specs))


@dataclass(frozen=True, init=False)
class ReleaseStackContext:
    """Validator-owned identities expected for one reviewed engine release."""

    runtime_digest: str
    base_engine_digest: str
    catalog_digest: str
    _catalog_json: bytes
    _target_specs: tuple[tuple[str, str], ...]

    def __init__(
        self,
        *,
        runtime_digest: str,
        base_engine_digest: str,
        catalog_snapshot: Mapping[str, object],
        catalog_digest: str,
        target_spec_digests: Mapping[str, str],
    ) -> None:
        object.__setattr__(
            self, "runtime_digest", _digest(runtime_digest, field="runtime_digest")
        )
        object.__setattr__(
            self,
            "base_engine_digest",
            _digest(base_engine_digest, field="base_engine_digest"),
        )
        catalog_json, checked_catalog_digest = _catalog_binding(
            catalog_snapshot, catalog_digest
        )
        object.__setattr__(self, "catalog_digest", checked_catalog_digest)
        object.__setattr__(self, "_catalog_json", catalog_json)
        object.__setattr__(
            self, "_target_specs", _context_spec_rows(catalog_json, target_spec_digests)
        )

    @property
    def catalog_snapshot(self) -> dict[str, Any]:
        return json.loads(self._catalog_json)

    @property
    def target_spec_digests(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self._target_specs))


def _validate_catalog_and_entries(
    *,
    manifest_catalog_json: bytes,
    manifest_catalog_digest: str,
    entries: tuple[tuple[str, ContributionRef], ...],
    context_catalog_json: bytes,
    context_catalog_digest: str,
    target_specs: tuple[tuple[str, str], ...],
) -> None:
    if manifest_catalog_digest != context_catalog_digest:
        raise StackManifestError("catalog digest does not match expected context")
    if manifest_catalog_json != context_catalog_json:
        raise StackManifestError("catalog snapshot does not match expected context")
    expected = dict(target_specs)
    for target, ref in entries:
        if target not in expected:
            raise StackManifestError(
                f"active target {target!r} is absent from expected catalog context"
            )
        if ref.target_spec_digest != expected[target]:
            raise StackManifestError(
                f"target spec digest for {target!r} does not match expected context"
            )


@dataclass(frozen=True, init=False)
class EvaluationStackManifest:
    """A complete, content-addressed hostile evaluation incumbent."""

    runtime_digest: str
    base_engine_digest: str
    arena_digest: str
    catalog_digest: str
    schema_version: int
    stack_policy_version: str
    _catalog_json: bytes
    _entries: tuple[tuple[str, ContributionRef], ...]

    def __init__(
        self,
        *,
        runtime_digest: str,
        base_engine_digest: str,
        arena_digest: str,
        catalog_snapshot: Mapping[str, object],
        catalog_digest: str,
        entries: Mapping[str, ContributionRef]
        | Iterable[tuple[str, ContributionRef]],
        schema_version: int = STACK_MANIFEST_SCHEMA_VERSION,
        stack_policy_version: str = EVALUATION_STACK_POLICY_VERSION,
    ) -> None:
        _current_version(
            schema_version,
            field="stack schema_version",
            expected=STACK_MANIFEST_SCHEMA_VERSION,
        )
        _current_version(
            stack_policy_version,
            field="stack_policy_version",
            expected=EVALUATION_STACK_POLICY_VERSION,
        )
        object.__setattr__(
            self, "runtime_digest", _digest(runtime_digest, field="runtime_digest")
        )
        object.__setattr__(
            self,
            "base_engine_digest",
            _digest(base_engine_digest, field="base_engine_digest"),
        )
        object.__setattr__(
            self, "arena_digest", _digest(arena_digest, field="arena_digest")
        )
        catalog_json, checked_catalog_digest = _catalog_binding(
            catalog_snapshot, catalog_digest
        )
        object.__setattr__(self, "catalog_digest", checked_catalog_digest)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "stack_policy_version", stack_policy_version)
        object.__setattr__(self, "_catalog_json", catalog_json)
        object.__setattr__(self, "_entries", _entry_rows(entries, integrated_only=False))

    @property
    def catalog_snapshot(self) -> dict[str, Any]:
        return json.loads(self._catalog_json)

    @property
    def entries(self) -> Mapping[str, ContributionRef]:
        return MappingProxyType(dict(self._entries))

    @property
    def digest(self) -> str:
        return canonical_digest("optima.stack.evaluation", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "arena_digest": self.arena_digest,
            "base_engine_digest": self.base_engine_digest,
            "catalog_digest": self.catalog_digest,
            "catalog_snapshot": self.catalog_snapshot,
            "entries": {target: ref.to_dict() for target, ref in self._entries},
            "runtime_digest": self.runtime_digest,
            "schema_version": self.schema_version,
            "stack_policy_version": self.stack_policy_version,
            "type": "evaluation_stack",
        }

    @classmethod
    def from_dict(cls, value: object) -> "EvaluationStackManifest":
        row = _strict_object(
            value,
            fields=frozenset(
                {
                    "arena_digest",
                    "base_engine_digest",
                    "catalog_digest",
                    "catalog_snapshot",
                    "entries",
                    "runtime_digest",
                    "schema_version",
                    "stack_policy_version",
                    "type",
                }
            ),
            name="evaluation stack manifest",
        )
        if row["type"] != "evaluation_stack":
            raise StackManifestError("evaluation stack type must be 'evaluation_stack'")
        if not isinstance(row["entries"], Mapping):
            raise StackManifestError("evaluation stack entries must be an object")
        entries = [
            (target, contribution_ref_from_dict(ref))
            for target, ref in row["entries"].items()
        ]
        return cls(
            runtime_digest=row["runtime_digest"],  # type: ignore[arg-type]
            base_engine_digest=row["base_engine_digest"],  # type: ignore[arg-type]
            arena_digest=row["arena_digest"],  # type: ignore[arg-type]
            catalog_snapshot=row["catalog_snapshot"],  # type: ignore[arg-type]
            catalog_digest=row["catalog_digest"],  # type: ignore[arg-type]
            entries=entries,
            schema_version=row["schema_version"],  # type: ignore[arg-type]
            stack_policy_version=row["stack_policy_version"],  # type: ignore[arg-type]
        )

    def validate_against(self, context: EvaluationStackContext) -> None:
        if not isinstance(context, EvaluationStackContext):
            raise TypeError("context must be an EvaluationStackContext")
        if self.runtime_digest != context.runtime_digest:
            raise StackManifestError("runtime digest does not match expected context")
        if self.base_engine_digest != context.base_engine_digest:
            raise StackManifestError("base engine digest does not match expected context")
        if self.arena_digest != context.arena_digest:
            raise StackManifestError("arena digest does not match expected context")
        _validate_catalog_and_entries(
            manifest_catalog_json=self._catalog_json,
            manifest_catalog_digest=self.catalog_digest,
            entries=self._entries,
            context_catalog_json=context._catalog_json,
            context_catalog_digest=context.catalog_digest,
            target_specs=context._target_specs,
        )

    def with_contribution(
        self,
        contribution: ContributionRef,
        *,
        remove: Iterable[str] = (),
    ) -> "EvaluationStackManifest":
        if not isinstance(contribution, (ProposalContributionRef, IntegratedContributionRef)):
            raise StackManifestError("contribution has invalid type")
        if isinstance(remove, (str, bytes)):
            raise StackManifestError("remove must be an iterable of target IDs")
        remove_ids = tuple(_target_id(item, field="remove target") for item in remove)
        if len(set(remove_ids)) != len(remove_ids):
            raise StackManifestError("remove contains duplicate targets")
        updated = dict(self._entries)
        for target in remove_ids:
            if target not in updated:
                raise StackManifestError(f"cannot remove inactive target {target!r}")
            del updated[target]
        updated[contribution.target_id] = contribution
        return EvaluationStackManifest(
            runtime_digest=self.runtime_digest,
            base_engine_digest=self.base_engine_digest,
            arena_digest=self.arena_digest,
            catalog_snapshot=self.catalog_snapshot,
            catalog_digest=self.catalog_digest,
            entries=updated,
            schema_version=self.schema_version,
            stack_policy_version=self.stack_policy_version,
        )


@dataclass(frozen=True, init=False)
class EngineReleaseManifest:
    """A chain-independent release identity containing reviewed source only."""

    runtime_digest: str
    base_engine_digest: str
    catalog_digest: str
    schema_version: int
    stack_policy_version: str
    _catalog_json: bytes
    _entries: tuple[tuple[str, IntegratedContributionRef], ...]

    def __init__(
        self,
        *,
        runtime_digest: str,
        base_engine_digest: str,
        catalog_snapshot: Mapping[str, object],
        catalog_digest: str,
        entries: Mapping[str, IntegratedContributionRef]
        | Iterable[tuple[str, IntegratedContributionRef]],
        schema_version: int = STACK_MANIFEST_SCHEMA_VERSION,
        stack_policy_version: str = ENGINE_RELEASE_POLICY_VERSION,
    ) -> None:
        _current_version(
            schema_version,
            field="stack schema_version",
            expected=STACK_MANIFEST_SCHEMA_VERSION,
        )
        _current_version(
            stack_policy_version,
            field="stack_policy_version",
            expected=ENGINE_RELEASE_POLICY_VERSION,
        )
        object.__setattr__(
            self, "runtime_digest", _digest(runtime_digest, field="runtime_digest")
        )
        object.__setattr__(
            self,
            "base_engine_digest",
            _digest(base_engine_digest, field="base_engine_digest"),
        )
        catalog_json, checked_catalog_digest = _catalog_binding(
            catalog_snapshot, catalog_digest
        )
        object.__setattr__(self, "catalog_digest", checked_catalog_digest)
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "stack_policy_version", stack_policy_version)
        object.__setattr__(self, "_catalog_json", catalog_json)
        object.__setattr__(self, "_entries", _entry_rows(entries, integrated_only=True))

    @property
    def catalog_snapshot(self) -> dict[str, Any]:
        return json.loads(self._catalog_json)

    @property
    def entries(self) -> Mapping[str, IntegratedContributionRef]:
        return MappingProxyType(dict(self._entries))

    @property
    def digest(self) -> str:
        return canonical_digest("optima.stack.release", self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "base_engine_digest": self.base_engine_digest,
            "catalog_digest": self.catalog_digest,
            "catalog_snapshot": self.catalog_snapshot,
            "entries": {target: ref.to_dict() for target, ref in self._entries},
            "runtime_digest": self.runtime_digest,
            "schema_version": self.schema_version,
            "stack_policy_version": self.stack_policy_version,
            "type": "engine_release",
        }

    @classmethod
    def from_dict(cls, value: object) -> "EngineReleaseManifest":
        row = _strict_object(
            value,
            fields=frozenset(
                {
                    "base_engine_digest",
                    "catalog_digest",
                    "catalog_snapshot",
                    "entries",
                    "runtime_digest",
                    "schema_version",
                    "stack_policy_version",
                    "type",
                }
            ),
            name="engine release manifest",
        )
        if row["type"] != "engine_release":
            raise StackManifestError("engine release type must be 'engine_release'")
        if not isinstance(row["entries"], Mapping):
            raise StackManifestError("engine release entries must be an object")
        entries: list[tuple[str, IntegratedContributionRef]] = []
        for target, raw_ref in row["entries"].items():
            ref = contribution_ref_from_dict(raw_ref)
            if not isinstance(ref, IntegratedContributionRef):
                raise StackManifestError(
                    "release manifests accept integrated contributions only"
                )
            entries.append((target, ref))
        return cls(
            runtime_digest=row["runtime_digest"],  # type: ignore[arg-type]
            base_engine_digest=row["base_engine_digest"],  # type: ignore[arg-type]
            catalog_snapshot=row["catalog_snapshot"],  # type: ignore[arg-type]
            catalog_digest=row["catalog_digest"],  # type: ignore[arg-type]
            entries=entries,
            schema_version=row["schema_version"],  # type: ignore[arg-type]
            stack_policy_version=row["stack_policy_version"],  # type: ignore[arg-type]
        )

    def validate_against(self, context: ReleaseStackContext) -> None:
        if not isinstance(context, ReleaseStackContext):
            raise TypeError("context must be a ReleaseStackContext")
        if self.runtime_digest != context.runtime_digest:
            raise StackManifestError("runtime digest does not match expected context")
        if self.base_engine_digest != context.base_engine_digest:
            raise StackManifestError("base engine digest does not match expected context")
        _validate_catalog_and_entries(
            manifest_catalog_json=self._catalog_json,
            manifest_catalog_digest=self.catalog_digest,
            entries=self._entries,
            context_catalog_json=context._catalog_json,
            context_catalog_digest=context.catalog_digest,
            target_specs=context._target_specs,
        )

    def validate_integrations(
        self,
        records: Mapping[str, IntegrationReviewRecord]
        | Iterable[tuple[str, IntegrationReviewRecord]],
    ) -> None:
        """Require one exact approved review record for every shipped target."""

        raw = tuple(records.items()) if isinstance(records, Mapping) else tuple(records)
        checked: dict[str, IntegrationReviewRecord] = {}
        for item in raw:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise StackManifestError("integration records must be target/record pairs")
            target, record = item
            target = _target_id(target, field="integration record target")
            if target in checked or type(record) is not IntegrationReviewRecord:
                raise StackManifestError("integration records are duplicated or untyped")
            if record.target_id != target:
                raise StackManifestError("integration record target differs from its key")
            checked[target] = record
        if set(checked) != set(self.entries):
            raise StackManifestError("release integration-record coverage differs")
        for target, ref in self.entries.items():
            checked[target].require_ref(ref)


def stack_manifest_from_dict(
    value: object,
) -> EvaluationStackManifest | EngineReleaseManifest:
    if not isinstance(value, Mapping):
        raise StackManifestError("stack manifest must be an object")
    kind = value.get("type")
    if kind == "evaluation_stack":
        return EvaluationStackManifest.from_dict(value)
    if kind == "engine_release":
        return EngineReleaseManifest.from_dict(value)
    raise StackManifestError(
        "stack manifest requires type 'evaluation_stack' or 'engine_release'"
    )
