"""Pure planning for one-target evaluation-stack transitions.

The types in this module describe immutable B/C/B-prime arms, sealed candidate
cohorts, and exact rollback reconstruction.  They do not launch engines,
interpret measurements, select winners, or mutate incumbent state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from optima.stack_identity import canonical_digest
from optima.stack_manifest import (
    ContributionRef,
    EvaluationStackContext,
    EvaluationStackManifest,
    IntegratedContributionRef,
    ProposalContributionRef,
)
from optima.target_catalog import TargetCatalog, TargetResolutionError
from optima._strict import require_digest


_PLAN_SCHEMA_VERSION = 1
_PLAN_POLICY_VERSION = "stack-plan.v1"


class StackPlanError(ValueError):
    """A requested stack transition is not one registered marginal delta."""


class StaleStackPlanError(StackPlanError):
    """A plan no longer applies to the supplied incumbent identity."""


def _digest(value: object, *, field: str) -> str:
    return require_digest(value, field=field, error=StackPlanError)


def _ref_dict(ref: ContributionRef) -> dict[str, object]:
    return ref.to_dict()


def _require_ref(value: object, *, field: str) -> ContributionRef:
    if not isinstance(value, (ProposalContributionRef, IntegratedContributionRef)):
        raise StackPlanError(f"{field} must be a contribution ref")
    return value


def _execution_order(
    arms: tuple[MarginalArmPlan, ...], entropy_digest: str
) -> tuple[str, ...]:
    return tuple(
        arm.selected_delta_digest
        for arm in sorted(
            arms,
            key=lambda arm: (
                canonical_digest(
                    "optima.stack.cohort-execution-key",
                    {
                        "entropy_digest": entropy_digest,
                        "selected_delta_digest": arm.selected_delta_digest,
                    },
                ),
                arm.selected_delta_digest,
            ),
        )
    )


@dataclass(frozen=True)
class StackArmIdentity:
    """Content identity of one complete materialized engine arm."""

    stack_digest: str
    tree_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "stack_digest",
            _digest(self.stack_digest, field="arm stack_digest"),
        )
        object.__setattr__(
            self,
            "tree_digest",
            _digest(self.tree_digest, field="arm tree_digest"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "stack_digest": self.stack_digest,
            "tree_digest": self.tree_digest,
        }


@dataclass(frozen=True)
class TargetTransition:
    """The exact registered target replacement represented by one C arm."""

    target_id: str
    target_spec_digest: str
    replacement: ContributionRef
    prior: ContributionRef | None
    displaced: tuple[ContributionRef, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.target_id, str) or not self.target_id:
            raise StackPlanError("transition target_id must be a non-empty string")
        object.__setattr__(
            self,
            "target_spec_digest",
            _digest(self.target_spec_digest, field="transition target_spec_digest"),
        )
        object.__setattr__(
            self, "replacement", _require_ref(self.replacement, field="replacement")
        )
        if self.prior is not None:
            object.__setattr__(self, "prior", _require_ref(self.prior, field="prior"))
        object.__setattr__(
            self,
            "displaced",
            tuple(_require_ref(ref, field="displaced entry") for ref in self.displaced),
        )
        if self.replacement.target_id != self.target_id:
            raise StackPlanError("replacement target does not match transition target")
        if self.replacement.target_spec_digest != self.target_spec_digest:
            raise StackPlanError("replacement target-spec digest does not match transition")
        if self.prior is not None:
            if self.prior.target_id != self.target_id:
                raise StackPlanError("prior contribution does not match transition target")
            if self.displaced:
                raise StackPlanError(
                    "same-target replacement cannot also displace active targets"
                )
            if self.prior.selected_delta_digest == self.replacement.selected_delta_digest:
                raise StackPlanError("same-target replacement has no executable delta")
        displaced_ids = tuple(ref.target_id for ref in self.displaced)
        if displaced_ids != tuple(sorted(displaced_ids)):
            raise StackPlanError("displaced contributions must be target-sorted")
        if len(set(displaced_ids)) != len(displaced_ids):
            raise StackPlanError("displaced contributions contain duplicate targets")
        if self.target_id in displaced_ids:
            raise StackPlanError("transition cannot displace its replacement target")

    @property
    def selected_delta_digest(self) -> str:
        return self.replacement.selected_delta_digest

    def to_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "target_spec_digest": self.target_spec_digest,
            "replacement": _ref_dict(self.replacement),
            "prior": None if self.prior is None else _ref_dict(self.prior),
            "displaced": [_ref_dict(ref) for ref in self.displaced],
        }


@dataclass(frozen=True)
class MarginalArmPlan:
    """One exact target transition over a frozen incumbent stack."""

    incumbent: EvaluationStackManifest
    candidate: EvaluationStackManifest
    transition: TargetTransition
    baseline_before: StackArmIdentity
    challenger: StackArmIdentity
    baseline_after: StackArmIdentity
    schema_version: int = _PLAN_SCHEMA_VERSION
    policy_version: str = _PLAN_POLICY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.incumbent, EvaluationStackManifest) or not isinstance(
            self.candidate, EvaluationStackManifest
        ):
            raise StackPlanError("marginal arm stacks must be evaluation manifests")
        if not isinstance(self.transition, TargetTransition):
            raise StackPlanError("marginal arm transition is invalid")
        if not all(
            isinstance(arm, StackArmIdentity)
            for arm in (self.baseline_before, self.challenger, self.baseline_after)
        ):
            raise StackPlanError("marginal arm identities are invalid")
        if type(self.schema_version) is not int or self.schema_version != _PLAN_SCHEMA_VERSION:
            raise StackPlanError("marginal arm schema_version must be 1")
        if self.policy_version != _PLAN_POLICY_VERSION:
            raise StackPlanError("marginal arm policy_version is unsupported")
        if self.baseline_before != self.baseline_after:
            raise StackPlanError("B and B-prime must bind the same exact incumbent")
        if self.baseline_before.stack_digest != self.incumbent.digest:
            raise StackPlanError("baseline arm does not bind incumbent stack")
        if self.challenger.stack_digest != self.candidate.digest:
            raise StackPlanError("challenger arm does not bind candidate stack")
        if self.challenger.tree_digest == self.baseline_before.tree_digest:
            raise StackPlanError("challenger and incumbent tree digests must differ")
        incumbent_entries = self.incumbent.entries
        candidate_entries = self.candidate.entries
        target_id = self.transition.target_id
        if incumbent_entries.get(target_id) != self.transition.prior:
            raise StackPlanError("transition prior does not match incumbent entry")
        if candidate_entries.get(target_id) != self.transition.replacement:
            raise StackPlanError("transition replacement does not match candidate entry")
        displaced = {
            ref.target_id: ref for ref in self.transition.displaced
        }
        if any(incumbent_entries.get(target) != ref for target, ref in displaced.items()):
            raise StackPlanError("transition displaced entries do not match incumbent")
        expected_targets = (set(incumbent_entries) - set(displaced)) | {target_id}
        if set(candidate_entries) != expected_targets:
            raise StackPlanError("candidate entries do not match transition target set")
        for active_id, incumbent_ref in incumbent_entries.items():
            if active_id not in displaced and active_id != target_id:
                if candidate_entries.get(active_id) != incumbent_ref:
                    raise StackPlanError(
                        f"candidate changed unrelated target {active_id!r}"
                    )

    @property
    def selected_delta_digest(self) -> str:
        return self.transition.selected_delta_digest

    @property
    def contribution_digest(self) -> str:
        return self.transition.replacement.digest

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "policy_version": self.policy_version,
            "incumbent_stack_digest": self.incumbent.digest,
            "candidate_stack_digest": self.candidate.digest,
            "transition": self.transition.to_dict(),
            "baseline_before": self.baseline_before.to_dict(),
            "challenger": self.challenger.to_dict(),
            "baseline_after": self.baseline_after.to_dict(),
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.stack.marginal-arm-plan", self.to_dict())

    def require_current(
        self,
        current: EvaluationStackManifest,
        *,
        tree_digest: str,
        expected_context: EvaluationStackContext,
    ) -> "MarginalArmPlan":
        current.validate_against(expected_context)
        current_tree = _digest(tree_digest, field="current tree_digest")
        if current.digest != self.incumbent.digest:
            raise StaleStackPlanError("marginal arm incumbent stack is stale")
        if current_tree != self.baseline_before.tree_digest:
            raise StaleStackPlanError("marginal arm incumbent tree is stale")
        return self

    def reopen(
        self,
        *,
        catalog: TargetCatalog,
        expected_context: EvaluationStackContext,
    ) -> "MarginalArmPlan":
        """Reconstruct and compare the complete registered transition."""

        expected = plan_marginal_arm(
            self.incumbent,
            self.transition.replacement,
            catalog=catalog,
            incumbent_tree_digest=self.baseline_before.tree_digest,
            candidate_tree_digest=self.challenger.tree_digest,
            expected_context=expected_context,
        )
        if expected.to_dict() != self.to_dict():
            raise StackPlanError("marginal arm does not reopen to its declared transition")
        return self


def _candidate_transition(
    incumbent: EvaluationStackManifest,
    replacement: ContributionRef,
    *,
    catalog: TargetCatalog,
    expected_context: EvaluationStackContext,
) -> tuple[EvaluationStackManifest, TargetTransition]:

    if not isinstance(incumbent, EvaluationStackManifest):
        raise TypeError("incumbent must be an EvaluationStackManifest")
    if not isinstance(catalog, TargetCatalog):
        raise TypeError("catalog must be a TargetCatalog")
    replacement = _require_ref(replacement, field="replacement")
    incumbent.validate_against(expected_context)
    if (
        catalog.digest != incumbent.catalog_digest
        or catalog.digest != expected_context.catalog_digest
        or catalog.snapshot() != incumbent.catalog_snapshot
        or catalog.snapshot() != expected_context.catalog_snapshot
    ):
        raise StackPlanError("planning catalog does not match the frozen stack context")
    target_id = replacement.target_id
    try:
        catalog.require(target_id)
        expected_spec = catalog.target_spec_digest(target_id)
        catalog.validate_active_targets(incumbent.entries)
    except TargetResolutionError as exc:
        raise StackPlanError(f"invalid registered transition: {exc}") from exc
    if replacement.target_spec_digest != expected_spec:
        raise StackPlanError(
            f"replacement target-spec digest is stale for {target_id!r}"
        )

    active = incumbent.entries
    prior = active.get(target_id)
    if prior is not None:
        remove: tuple[str, ...] = ()
        displaced: tuple[ContributionRef, ...] = ()
    else:
        # Directional displacement cannot be inverted.  In particular, an
        # atomic incumbent is never decomposed by proposing one member.
        blockers = tuple(
            sorted(
                active_id
                for active_id in active
                if target_id in catalog.displacement_closure(active_id)
            )
        )
        if blockers:
            raise StackPlanError(
                f"target {target_id!r} cannot implicitly decompose active "
                f"targets {blockers!r}"
            )
        remove = tuple(
            sorted(set(active) & set(catalog.displacement_closure(target_id)))
        )
        displaced = tuple(active[target] for target in remove)

    transition = TargetTransition(
        target_id=target_id,
        target_spec_digest=expected_spec,
        replacement=replacement,
        prior=prior,
        displaced=displaced,
    )
    try:
        candidate = incumbent.with_contribution(replacement, remove=remove)
        catalog.validate_active_targets(candidate.entries)
        candidate.validate_against(expected_context)
    except (ValueError, TargetResolutionError) as exc:
        raise StackPlanError(f"invalid marginal transition: {exc}") from exc

    expected_targets = (set(active) - set(remove)) | {target_id}
    if set(candidate.entries) != expected_targets:
        raise StackPlanError("candidate changed entries outside the target transition")
    for active_id, incumbent_ref in active.items():
        if active_id not in remove and active_id != target_id:
            if candidate.entries.get(active_id) != incumbent_ref:
                raise StackPlanError(
                    f"candidate changed unrelated target {active_id!r}"
                )
    if candidate.digest == incumbent.digest:
        raise StackPlanError("marginal transition does not change the stack")
    return candidate, transition


def plan_candidate_stack(
    incumbent: EvaluationStackManifest,
    replacement: ContributionRef,
    *,
    catalog: TargetCatalog,
    expected_context: EvaluationStackContext,
) -> EvaluationStackManifest:
    """Construct the exact C stack before its engine tree is materialized."""

    candidate, _ = _candidate_transition(
        incumbent,
        replacement,
        catalog=catalog,
        expected_context=expected_context,
    )
    return candidate


def plan_marginal_arm(
    incumbent: EvaluationStackManifest,
    replacement: ContributionRef,
    *,
    catalog: TargetCatalog,
    incumbent_tree_digest: str,
    candidate_tree_digest: str,
    expected_context: EvaluationStackContext,
) -> MarginalArmPlan:
    """Bind an exact C transition to independently materialized tree identities."""

    candidate, transition = _candidate_transition(
        incumbent,
        replacement,
        catalog=catalog,
        expected_context=expected_context,
    )

    incumbent_tree = _digest(
        incumbent_tree_digest, field="incumbent tree_digest"
    )
    candidate_tree = _digest(candidate_tree_digest, field="candidate tree_digest")
    return MarginalArmPlan(
        incumbent=incumbent,
        candidate=candidate,
        transition=transition,
        baseline_before=StackArmIdentity(incumbent.digest, incumbent_tree),
        challenger=StackArmIdentity(candidate.digest, candidate_tree),
        baseline_after=StackArmIdentity(incumbent.digest, incumbent_tree),
    )


@dataclass(frozen=True)
class RollbackPlan:
    """Pure reconstruction of the exact stack preceding one marginal arm."""

    expected_current: StackArmIdentity
    restored: StackArmIdentity
    restored_manifest: EvaluationStackManifest
    source_arm_digest: str
    schema_version: int = _PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.expected_current, StackArmIdentity) or not isinstance(
            self.restored, StackArmIdentity
        ):
            raise StackPlanError("rollback stack identities are invalid")
        if not isinstance(self.restored_manifest, EvaluationStackManifest):
            raise StackPlanError("rollback restored manifest is invalid")
        if type(self.schema_version) is not int or self.schema_version != _PLAN_SCHEMA_VERSION:
            raise StackPlanError("rollback schema_version must be 1")
        object.__setattr__(
            self,
            "source_arm_digest",
            _digest(self.source_arm_digest, field="rollback source_arm_digest"),
        )
        if self.restored.stack_digest != self.restored_manifest.digest:
            raise StackPlanError("rollback manifest does not match restored stack")
        if self.expected_current == self.restored:
            raise StackPlanError("rollback must restore a different whole stack")

    @classmethod
    def from_arm(
        cls,
        arm: MarginalArmPlan,
        *,
        catalog: TargetCatalog,
        expected_context: EvaluationStackContext,
    ) -> "RollbackPlan":
        arm.reopen(catalog=catalog, expected_context=expected_context)
        return cls(
            expected_current=arm.challenger,
            restored=StackArmIdentity(
                arm.baseline_before.stack_digest,
                arm.baseline_before.tree_digest,
            ),
            restored_manifest=arm.incumbent,
            source_arm_digest=arm.digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "expected_current": self.expected_current.to_dict(),
            "restored": self.restored.to_dict(),
            "restored_manifest_digest": self.restored_manifest.digest,
            "source_arm_digest": self.source_arm_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.stack.rollback-plan", self.to_dict())

    def reopen(
        self,
        source_arm: MarginalArmPlan,
        *,
        catalog: TargetCatalog,
        expected_context: EvaluationStackContext,
    ) -> "RollbackPlan":
        expected = RollbackPlan.from_arm(
            source_arm,
            catalog=catalog,
            expected_context=expected_context,
        )
        if expected.to_dict() != self.to_dict():
            raise StackPlanError("rollback does not reopen to its source arm")
        return self

    def reconstruct(
        self,
        current: EvaluationStackManifest,
        *,
        tree_digest: str,
        source_arm: MarginalArmPlan,
        catalog: TargetCatalog,
        expected_context: EvaluationStackContext,
    ) -> tuple[EvaluationStackManifest, str]:
        self.reopen(
            source_arm,
            catalog=catalog,
            expected_context=expected_context,
        )
        current.validate_against(expected_context)
        current_tree = _digest(tree_digest, field="current tree_digest")
        if current.digest != self.expected_current.stack_digest:
            raise StaleStackPlanError("rollback current stack is stale")
        if current_tree != self.expected_current.tree_digest:
            raise StaleStackPlanError("rollback current tree is stale")
        self.restored_manifest.validate_against(expected_context)
        return self.restored_manifest, self.restored.tree_digest


@dataclass(frozen=True)
class CohortPlan:
    """A finite set of C arms sealed against one exact incumbent."""

    incumbent: StackArmIdentity
    arms: tuple[MarginalArmPlan, ...]
    entropy_digest: str
    authority_order: tuple[ContributionRef, ...]
    execution_order: tuple[str, ...]
    schema_version: int = _PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.incumbent, StackArmIdentity):
            raise StackPlanError("cohort incumbent identity is invalid")
        if type(self.schema_version) is not int or self.schema_version != _PLAN_SCHEMA_VERSION:
            raise StackPlanError("cohort schema_version must be 1")
        object.__setattr__(self, "arms", tuple(self.arms))
        object.__setattr__(self, "authority_order", tuple(self.authority_order))
        object.__setattr__(self, "execution_order", tuple(self.execution_order))
        object.__setattr__(
            self,
            "entropy_digest",
            _digest(self.entropy_digest, field="cohort entropy_digest"),
        )
        if not self.arms:
            raise StackPlanError("cohort must contain at least one candidate arm")
        if not all(isinstance(arm, MarginalArmPlan) for arm in self.arms):
            raise StackPlanError("cohort arms must be MarginalArmPlan values")
        canonical_deltas = tuple(arm.selected_delta_digest for arm in self.arms)
        if canonical_deltas != tuple(sorted(canonical_deltas)):
            raise StackPlanError("cohort arms must be selected-delta sorted")
        if len(set(canonical_deltas)) != len(canonical_deltas):
            raise StackPlanError("cohort contains duplicate selected deltas")
        if any(arm.baseline_before != self.incumbent for arm in self.arms):
            raise StackPlanError("cohort arms do not share the frozen incumbent")
        candidate_stacks = tuple(arm.challenger.stack_digest for arm in self.arms)
        candidate_trees = tuple(arm.challenger.tree_digest for arm in self.arms)
        if len(set(candidate_stacks)) != len(candidate_stacks):
            raise StackPlanError("cohort contains duplicate candidate stacks")
        if len(set(candidate_trees)) != len(candidate_trees):
            raise StackPlanError("cohort contains duplicate candidate trees")

        object.__setattr__(
            self,
            "authority_order",
            tuple(
                _require_ref(ref, field="authority order entry")
                for ref in self.authority_order
            ),
        )
        arm_refs = tuple(arm.contribution_digest for arm in self.arms)
        authority_refs = tuple(ref.digest for ref in self.authority_order)
        if len(set(authority_refs)) != len(authority_refs):
            raise StackPlanError("authority order contains duplicate contributions")
        if set(authority_refs) != set(arm_refs) or len(authority_refs) != len(arm_refs):
            raise StackPlanError(
                "authority order must contain every cohort contribution exactly once"
            )
        if set(self.execution_order) != set(canonical_deltas) or len(
            self.execution_order
        ) != len(canonical_deltas):
            raise StackPlanError(
                "execution order must contain every selected delta exactly once"
            )
        if self.execution_order != _execution_order(self.arms, self.entropy_digest):
            raise StackPlanError("execution order does not match sealed entropy")

    @classmethod
    def seal(
        cls,
        arms: Iterable[MarginalArmPlan],
        *,
        entropy_digest: str,
        authority_order: Iterable[ContributionRef],
        catalog: TargetCatalog,
        expected_context: EvaluationStackContext,
    ) -> "CohortPlan":
        raw_arms = tuple(arms)
        if not raw_arms:
            raise StackPlanError("cohort must contain at least one candidate arm")
        for arm in raw_arms:
            if not isinstance(arm, MarginalArmPlan):
                raise StackPlanError("cohort arms must be MarginalArmPlan values")
            arm.reopen(catalog=catalog, expected_context=expected_context)
        canonical_arms = tuple(
            sorted(raw_arms, key=lambda arm: arm.selected_delta_digest)
        )
        entropy = _digest(entropy_digest, field="cohort entropy_digest")
        execution = _execution_order(canonical_arms, entropy)
        first = canonical_arms[0]
        return cls(
            incumbent=StackArmIdentity(
                first.baseline_before.stack_digest,
                first.baseline_before.tree_digest,
            ),
            arms=canonical_arms,
            entropy_digest=entropy,
            authority_order=tuple(
                _require_ref(ref, field="authority order entry")
                for ref in authority_order
            ),
            execution_order=execution,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "incumbent": self.incumbent.to_dict(),
            "arms": [arm.digest for arm in self.arms],
            "entropy_digest": self.entropy_digest,
            "authority_order": [ref.digest for ref in self.authority_order],
            "execution_order": list(self.execution_order),
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.stack.cohort-plan", self.to_dict())

    @property
    def execution_arms(self) -> tuple[MarginalArmPlan, ...]:
        by_delta = {arm.selected_delta_digest: arm for arm in self.arms}
        return tuple(by_delta[digest] for digest in self.execution_order)

    @property
    def authority_arms(self) -> tuple[MarginalArmPlan, ...]:
        by_ref = {arm.contribution_digest: arm for arm in self.arms}
        return tuple(by_ref[ref.digest] for ref in self.authority_order)

    def require_current(
        self,
        current: EvaluationStackManifest,
        *,
        tree_digest: str,
        expected_context: EvaluationStackContext,
    ) -> "CohortPlan":
        current.validate_against(expected_context)
        current_tree = _digest(tree_digest, field="current tree_digest")
        if current.digest != self.incumbent.stack_digest:
            raise StaleStackPlanError("cohort incumbent stack is stale")
        if current_tree != self.incumbent.tree_digest:
            raise StaleStackPlanError("cohort incumbent tree is stale")
        return self

    def reopen(
        self,
        *,
        catalog: TargetCatalog,
        expected_context: EvaluationStackContext,
    ) -> "CohortPlan":
        expected = CohortPlan.seal(
            self.arms,
            entropy_digest=self.entropy_digest,
            authority_order=self.authority_order,
            catalog=catalog,
            expected_context=expected_context,
        )
        if expected.to_dict() != self.to_dict():
            raise StackPlanError("cohort does not reopen to its declared arms and order")
        return self
