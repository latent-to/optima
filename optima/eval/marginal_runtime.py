"""Bind marginal stack plans to the generic isolated engine executor.

This module is deliberately only identity and lifecycle plumbing.  It derives
every candidate launch from one incumbent launch, derives every session plan
from one incumbent workload, executes ``B,C1..Ck,B'`` under one absolute
deadline, and returns raw execution evidence.  It does not score, retry,
qualify, select, crown, or mutate a stack.

``MaterializedArmBinding`` is a trusted host-local value.  Its tree must be the
live result returned by the validator materializer, not a miner-supplied or
deserialized assertion.  Finalized hostile intake and durable worker-tree
publication remain a separate control-plane boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from optima.engine_tree import MaterializedEngineTree
from optima.eval.engine_launch import (
    EngineLaunchError,
    EngineLaunchSpec,
    TrustedLaunchBinding,
    resolve_engine_launch,
)
from optima.eval.oci_backend import (
    EngineExecutionEvidence,
    TrustedArenaModelMountReceipt,
    expected_runtime_preflight,
    runtime_identity_from_preflight,
)
from optima.eval.oci_outer_session import SessionExecutionPlan
from optima.eval.runtime_preflight import RuntimePreflightReceipt
from optima.discovery import DiscoveryArmPlan, reopen_discovery_engine_binding
from optima.discovery_overlay import DiscoveryActivationReceipt
from optima.stack_identity import require_sha256_hex
from optima.stack_manifest import (
    EvaluationStackContext,
    EvaluationStackManifest,
    IntegratedContributionRef,
    ProposalContributionRef,
)
from optima.stack_plan import CohortPlan, MarginalArmPlan, StackPlanError
from optima.target_catalog import TargetCatalog


_TREE_METADATA = "metadata/optima_engine_tree.json"


class MarginalRuntimeError(ValueError):
    """A marginal plan cannot be bound to one exact runtime lifecycle."""


ExecutableArm = MarginalArmPlan | DiscoveryArmPlan
RuntimeSource = MarginalArmPlan | CohortPlan | DiscoveryArmPlan


class EngineExecutor(Protocol):
    """The raw PR2 executor interface consumed by this bridge."""

    def execute(
        self,
        launch: EngineLaunchSpec,
        binding: TrustedLaunchBinding,
        mount: TrustedArenaModelMountReceipt,
        plan: SessionExecutionPlan,
        *,
        deadline: float,
    ) -> EngineExecutionEvidence: ...


def _digest(value: object, *, field: str) -> str:
    try:
        result = require_sha256_hex(value, field=field)
    except ValueError as exc:
        raise MarginalRuntimeError(str(exc)) from exc
    return result


def _native_environment(binding: TrustedLaunchBinding) -> dict[str, object]:
    row = binding.native_build_spec.to_dict()
    row.pop("tree_digest")
    return row


def _expected_contributions(
    stack: EvaluationStackManifest,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for target_id, ref in sorted(stack.entries.items()):
        if type(ref) is ProposalContributionRef:
            source_kind = "proposal_artifact"
            source_digest = ref.artifact_digest
        elif type(ref) is IntegratedContributionRef:
            source_kind = "integrated_source"
            source_digest = ref.integrated_source_tree_digest
        else:  # pragma: no cover - EvaluationStackManifest is already closed
            raise MarginalRuntimeError("stack contains an unsupported contribution ref")
        rows.append(
            {
                "contribution_ref_digest": ref.digest,
                "namespace": f"optima_c_{ref.selected_delta_digest}",
                "selected_delta_digest": ref.selected_delta_digest,
                "selected_payload_digest": ref.selected_payload_digest,
                "source_digest": source_digest,
                "source_kind": source_kind,
                "target_id": target_id,
                "target_spec_digest": ref.target_spec_digest,
            }
        )
    return rows


def _tree_metadata(tree: MaterializedEngineTree) -> dict[str, object]:
    metadata_row = next((row for row in tree.files if row.path == _TREE_METADATA), None)
    if metadata_row is None:
        raise MarginalRuntimeError("materialized tree lacks its metadata inventory")
    path = tree.root / _TREE_METADATA
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise MarginalRuntimeError(f"cannot reopen materialized tree metadata: {exc}") from None
    if (
        len(data) != metadata_row.size
        or hashlib.sha256(data).hexdigest() != metadata_row.sha256
    ):
        raise MarginalRuntimeError("materialized tree metadata differs from trusted inventory")
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise MarginalRuntimeError(f"materialized tree metadata is invalid: {exc}") from None
    if not isinstance(value, dict):
        raise MarginalRuntimeError("materialized tree metadata must be an object")
    return value


def _require_tree_stack(
    tree: MaterializedEngineTree,
    stack: EvaluationStackManifest,
    *,
    expected_tree_digest: str,
) -> None:
    """Bind a live trusted materializer result to its exact stack inventory."""

    if type(tree) is not MaterializedEngineTree:
        raise MarginalRuntimeError("tree binding is not a MaterializedEngineTree")
    expected_tree = _digest(expected_tree_digest, field="expected tree_digest")
    if tree.stack_digest != stack.digest or tree.tree_digest != expected_tree:
        raise MarginalRuntimeError("materialized tree identity differs from its stack arm")
    metadata = _tree_metadata(tree)
    if metadata.get("stack_digest") != stack.digest:
        raise MarginalRuntimeError("materialized tree metadata names another stack")
    if metadata.get("contributions") != _expected_contributions(stack):
        raise MarginalRuntimeError(
            "materialized tree contribution inventory differs from its stack manifest"
        )
    expected_manifest = "manifest.toml" if stack.entries else None
    if tree.runtime_manifest != expected_manifest or metadata.get(
        "runtime_manifest"
    ) != expected_manifest:
        raise MarginalRuntimeError("materialized tree runtime manifest differs from its stack")


@dataclass(frozen=True)
class MaterializedArmBinding:
    """Trusted in-process materializer output plus its host-local launch binding."""

    tree: MaterializedEngineTree
    launch_binding: TrustedLaunchBinding

    def __post_init__(self) -> None:
        if type(self.tree) is not MaterializedEngineTree:
            raise MarginalRuntimeError("tree must be an exact MaterializedEngineTree")
        if type(self.launch_binding) is not TrustedLaunchBinding:
            raise MarginalRuntimeError("launch_binding must be a TrustedLaunchBinding")


@dataclass(frozen=True)
class PreparedCandidateRuntime:
    """One C arm bound to a mechanically derived launch and common workload."""

    arm: ExecutableArm
    binding: MaterializedArmBinding
    launch: EngineLaunchSpec
    session_plan: SessionExecutionPlan

    def __post_init__(self) -> None:
        if type(self.arm) not in {MarginalArmPlan, DiscoveryArmPlan}:
            raise MarginalRuntimeError("candidate arm has an unsupported type")
        if type(self.binding) is not MaterializedArmBinding:
            raise MarginalRuntimeError("candidate binding has the wrong type")
        if type(self.launch) is not EngineLaunchSpec:
            raise MarginalRuntimeError("candidate launch has the wrong type")
        if type(self.session_plan) is not SessionExecutionPlan:
            raise MarginalRuntimeError("candidate session plan has the wrong type")
        if (
            self.launch.stack_digest != self.arm.challenger.stack_digest
            or self.launch.tree_digest != self.arm.challenger.tree_digest
            or self.session_plan.launch_digest != self.launch.digest
        ):
            raise MarginalRuntimeError("candidate runtime does not bind its marginal arm")

@dataclass(frozen=True)
class PreparedMarginalRuntime:
    """A completely validated B,C1..Ck,B-prime runtime lifecycle."""

    source: RuntimeSource
    incumbent_binding: MaterializedArmBinding
    baseline_launch: EngineLaunchSpec
    baseline_session_plan: SessionExecutionPlan
    candidates: tuple[PreparedCandidateRuntime, ...]

    def __post_init__(self) -> None:
        if type(self.source) not in {MarginalArmPlan, CohortPlan, DiscoveryArmPlan}:
            raise MarginalRuntimeError("runtime source has an unsupported type")
        if type(self.incumbent_binding) is not MaterializedArmBinding:
            raise MarginalRuntimeError("incumbent binding has the wrong type")
        if type(self.baseline_launch) is not EngineLaunchSpec:
            raise MarginalRuntimeError("baseline launch has the wrong type")
        if type(self.baseline_session_plan) is not SessionExecutionPlan:
            raise MarginalRuntimeError("baseline session plan has the wrong type")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.candidates or any(
            type(candidate) is not PreparedCandidateRuntime
            for candidate in self.candidates
        ):
            raise MarginalRuntimeError("prepared runtime requires typed candidates")
        expected = (
            (self.source,)
            if type(self.source) in {MarginalArmPlan, DiscoveryArmPlan}
            else self.source.execution_arms
        )
        if tuple(candidate.arm for candidate in self.candidates) != expected:
            raise MarginalRuntimeError("candidate order differs from the sealed runtime source")
        if (
            self.baseline_launch.stack_digest
            != self.candidates[0].arm.baseline_before.stack_digest
            or self.baseline_launch.tree_digest
            != self.candidates[0].arm.baseline_before.tree_digest
        ):
            raise MarginalRuntimeError("baseline launch differs from the frozen incumbent")
        if self.baseline_session_plan.launch_digest != self.baseline_launch.digest:
            raise MarginalRuntimeError("baseline session plan names another launch")
        _validate_prepared_runtime(self)

@dataclass(frozen=True)
class CandidateLifecycleEvidence:
    """Raw C execution bound to its trusted arm label outside engine identity."""

    candidate: PreparedCandidateRuntime
    execution: EngineExecutionEvidence

    def __post_init__(self) -> None:
        if type(self.candidate) is not PreparedCandidateRuntime:
            raise MarginalRuntimeError("candidate evidence binding has the wrong type")
        if type(self.execution) is not EngineExecutionEvidence:
            raise MarginalRuntimeError("candidate execution evidence has the wrong type")
        if self.execution.launch_digest != self.candidate.launch.digest:
            raise MarginalRuntimeError("candidate evidence launch binding is invalid")

    @property
    def arm(self) -> ExecutableArm:
        return self.candidate.arm

@dataclass(frozen=True)
class MarginalLifecycleEvidence:
    """Complete raw lifecycle facts; deliberately has no score or canonical verdict."""

    prepared: PreparedMarginalRuntime
    baseline_before: EngineExecutionEvidence
    candidates: tuple[CandidateLifecycleEvidence, ...]
    baseline_after: EngineExecutionEvidence

    def __post_init__(self) -> None:
        if type(self.prepared) is not PreparedMarginalRuntime:
            raise MarginalRuntimeError("lifecycle prepared binding has the wrong type")
        if type(self.baseline_before) is not EngineExecutionEvidence or type(
            self.baseline_after
        ) is not EngineExecutionEvidence:
            raise MarginalRuntimeError("baseline lifecycle evidence has the wrong type")
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not self.candidates or any(
            type(row) is not CandidateLifecycleEvidence for row in self.candidates
        ):
            raise MarginalRuntimeError("lifecycle candidates are invalid")
        if (
            self.baseline_before.launch_digest != self.prepared.baseline_launch.digest
            or self.baseline_after.launch_digest != self.prepared.baseline_launch.digest
            or tuple(row.candidate for row in self.candidates)
            != self.prepared.candidates
        ):
            raise MarginalRuntimeError("lifecycle evidence was relabeled or reordered")

    @property
    def source(self) -> RuntimeSource:
        return self.prepared.source

def _require_context(
    launch: EngineLaunchSpec, expected_context: EvaluationStackContext
) -> None:
    if type(expected_context) is not EvaluationStackContext:
        raise MarginalRuntimeError("expected_context has the wrong type")
    observed = (
        launch.runtime_digest,
        launch.base_engine_digest,
        launch.arena_digest,
    )
    expected = (
        expected_context.runtime_digest,
        expected_context.base_engine_digest,
        expected_context.arena_digest,
    )
    if observed != expected:
        raise MarginalRuntimeError("baseline launch differs from the frozen stack context")


def _require_resolved_tree(
    launch: EngineLaunchSpec,
    binding: MaterializedArmBinding,
) -> None:
    try:
        resolved = resolve_engine_launch(launch, binding.launch_binding)
    except (EngineLaunchError, OSError, TypeError, ValueError) as exc:
        raise MarginalRuntimeError(f"engine launch binding failed: {exc}") from None
    try:
        trusted_root = binding.tree.root.resolve(strict=True)
        resolved_root = resolved.materialized_tree.root.resolve(strict=True)
    except OSError as exc:
        raise MarginalRuntimeError(f"materialized tree root is unavailable: {exc}") from None
    if trusted_root != resolved_root or (
        resolved.materialized_tree.stack_digest,
        resolved.materialized_tree.tree_digest,
        resolved.materialized_tree.files,
        resolved.materialized_tree.runtime_manifest,
    ) != (
        binding.tree.stack_digest,
        binding.tree.tree_digest,
        binding.tree.files,
        binding.tree.runtime_manifest,
    ):
        raise MarginalRuntimeError(
            "reopened engine tree differs from the trusted materializer result"
        )


def _resolve_materialized_binding(
    launch: EngineLaunchSpec,
    binding: MaterializedArmBinding,
    stack: EvaluationStackManifest,
    *,
    expected_tree_digest: str,
) -> None:
    _require_tree_stack(
        binding.tree,
        stack,
        expected_tree_digest=expected_tree_digest,
    )
    _require_resolved_tree(launch, binding)


def _require_discovery_tree(
    launch: EngineLaunchSpec,
    binding: MaterializedArmBinding,
    arm: DiscoveryArmPlan,
) -> None:
    tree = binding.tree
    if (
        tree.stack_digest != arm.candidate_stack_digest
        or tree.tree_digest != arm.candidate_tree_digest
    ):
        raise MarginalRuntimeError("discovery tree differs from its candidate arm")
    try:
        discovery = reopen_discovery_engine_binding(tree)
    except (OSError, TypeError, ValueError) as exc:
        raise MarginalRuntimeError(
            f"discovery engine tree failed to reopen: {exc}"
        ) from None
    if (
        discovery.materialized_tree != tree
        or discovery.incumbent_stack_digest != arm.incumbent.digest
        or discovery.incumbent_tree_digest != arm.incumbent_tree_digest
        or discovery.discovery.proposal_digest != arm.proposal_digest
        or discovery.policy.digest != arm.policy_digest
        or discovery.build_profile.digest != arm.build_profile_digest
    ):
        raise MarginalRuntimeError("discovery tree metadata differs from its arm")
    metadata = _tree_metadata(tree)
    expected_manifest = "manifest.toml" if arm.incumbent.entries else None
    if (
        metadata.get("stack_digest") != arm.candidate_stack_digest
        or metadata.get("contributions") != _expected_contributions(arm.incumbent)
        or metadata.get("runtime_manifest") != expected_manifest
        or tree.runtime_manifest != expected_manifest
    ):
        raise MarginalRuntimeError(
            "discovery tree changed the incumbent contribution inventory"
        )
    _require_resolved_tree(launch, binding)


def _require_baseline_session(
    launch: EngineLaunchSpec,
    binding: TrustedLaunchBinding,
    plan: SessionExecutionPlan,
) -> None:
    if type(plan) is not SessionExecutionPlan:
        raise MarginalRuntimeError("baseline session plan has the wrong type")
    receipt = binding.runtime_preflight_receipt
    if type(receipt) is not RuntimePreflightReceipt:
        raise MarginalRuntimeError("baseline binding lacks a typed runtime preflight")
    expected = expected_runtime_preflight(launch, receipt)
    if (
        plan.launch_digest != launch.digest
        or plan.expected_engine_config_digest != launch.engine_config_digest
        or plan.engine_config.digest != launch.engine_config_digest
        or plan.engine_config.tp_size != launch.hardware.tp_size
        or plan.expected_preflight != expected
        or plan.expected_discovery_overlay_identity_digest is not None
    ):
        raise MarginalRuntimeError("baseline session plan differs from its launch")


def _candidate_runtime(
    arm: ExecutableArm,
    *,
    baseline_launch: EngineLaunchSpec,
    baseline_binding: MaterializedArmBinding,
    baseline_session: SessionExecutionPlan,
    candidate_binding: MaterializedArmBinding,
) -> PreparedCandidateRuntime:
    base_local = baseline_binding.launch_binding
    candidate_local = candidate_binding.launch_binding
    if (
        candidate_local.controller_distribution_digest
        != base_local.controller_distribution_digest
        or candidate_local.runtime_preflight_receipt
        != base_local.runtime_preflight_receipt
        or candidate_local.physical_hardware != base_local.physical_hardware
        or _native_environment(candidate_local) != _native_environment(base_local)
    ):
        raise MarginalRuntimeError(
            "candidate changed controller, runtime preflight, or native-build environment"
        )
    candidate_launch = replace(
        baseline_launch,
        stack_digest=arm.challenger.stack_digest,
        tree_digest=arm.challenger.tree_digest,
        native_build_spec_digest=candidate_local.native_build_spec.digest,
    )
    if type(arm) is MarginalArmPlan:
        _resolve_materialized_binding(
            candidate_launch,
            candidate_binding,
            arm.candidate,
            expected_tree_digest=arm.challenger.tree_digest,
        )
    else:
        _require_discovery_tree(candidate_launch, candidate_binding, arm)
    receipt = candidate_local.runtime_preflight_receipt
    if type(receipt) is not RuntimePreflightReceipt:
        raise MarginalRuntimeError("candidate lacks a typed runtime preflight")
    candidate_preflight = expected_runtime_preflight(candidate_launch, receipt)
    candidate_session = replace(
        baseline_session,
        launch_digest=candidate_launch.digest,
        expected_preflight=candidate_preflight,
        expected_discovery_overlay_identity_digest=(
            arm.overlay_identity_digest
            if type(arm) is DiscoveryArmPlan
            else None
        ),
    )
    return PreparedCandidateRuntime(
        arm,
        candidate_binding,
        candidate_launch,
        candidate_session,
    )


def _validate_prepared_runtime(prepared: PreparedMarginalRuntime) -> None:
    first = prepared.candidates[0].arm
    _resolve_materialized_binding(
        prepared.baseline_launch,
        prepared.incumbent_binding,
        first.incumbent,
        expected_tree_digest=first.baseline_before.tree_digest,
    )
    _require_baseline_session(
        prepared.baseline_launch,
        prepared.incumbent_binding.launch_binding,
        prepared.baseline_session_plan,
    )
    for candidate in prepared.candidates:
        expected = _candidate_runtime(
            candidate.arm,
            baseline_launch=prepared.baseline_launch,
            baseline_binding=prepared.incumbent_binding,
            baseline_session=prepared.baseline_session_plan,
            candidate_binding=candidate.binding,
        )
        if candidate != expected:
            raise MarginalRuntimeError(
                "prepared candidate changed the common launch or workload"
            )


def _prepare(
    source: RuntimeSource,
    arms: tuple[ExecutableArm, ...],
    *,
    expected_context: EvaluationStackContext,
    incumbent_launch: EngineLaunchSpec,
    incumbent_binding: MaterializedArmBinding,
    candidate_bindings: Mapping[str, MaterializedArmBinding],
    baseline_session_plan: SessionExecutionPlan,
) -> PreparedMarginalRuntime:
    if type(incumbent_launch) is not EngineLaunchSpec:
        raise MarginalRuntimeError("incumbent_launch has the wrong type")
    if type(incumbent_binding) is not MaterializedArmBinding:
        raise MarginalRuntimeError("incumbent_binding has the wrong type")
    if not isinstance(candidate_bindings, Mapping):
        raise MarginalRuntimeError("candidate_bindings must be a mapping")
    _require_context(incumbent_launch, expected_context)
    first = arms[0]
    if (
        incumbent_launch.stack_digest != first.baseline_before.stack_digest
        or incumbent_launch.tree_digest != first.baseline_before.tree_digest
    ):
        raise MarginalRuntimeError("incumbent launch differs from the frozen baseline arm")
    expected_keys = {arm.selected_delta_digest for arm in arms}
    if set(candidate_bindings) != expected_keys or any(
        not isinstance(key, str) for key in candidate_bindings
    ):
        raise MarginalRuntimeError(
            "candidate bindings must cover every selected delta exactly once"
        )
    candidates = tuple(
        _candidate_runtime(
            arm,
            baseline_launch=incumbent_launch,
            baseline_binding=incumbent_binding,
            baseline_session=baseline_session_plan,
            candidate_binding=candidate_bindings[arm.selected_delta_digest],
        )
        for arm in arms
    )
    return PreparedMarginalRuntime(
        source,
        incumbent_binding,
        incumbent_launch,
        baseline_session_plan,
        candidates,
    )


def prepare_marginal_runtime(
    arm: MarginalArmPlan,
    *,
    catalog: TargetCatalog,
    expected_context: EvaluationStackContext,
    incumbent_launch: EngineLaunchSpec,
    incumbent_binding: MaterializedArmBinding,
    candidate_binding: MaterializedArmBinding,
    baseline_session_plan: SessionExecutionPlan,
) -> PreparedMarginalRuntime:
    """Reopen one marginal arm and bind its B,C,B-prime runtime inputs."""

    if type(arm) is not MarginalArmPlan or type(catalog) is not TargetCatalog:
        raise MarginalRuntimeError("arm or catalog has the wrong type")
    try:
        arm.reopen(catalog=catalog, expected_context=expected_context)
    except (StackPlanError, ValueError, TypeError) as exc:
        raise MarginalRuntimeError(f"marginal arm is stale or invalid: {exc}") from None
    return _prepare(
        arm,
        (arm,),
        expected_context=expected_context,
        incumbent_launch=incumbent_launch,
        incumbent_binding=incumbent_binding,
        candidate_bindings={arm.selected_delta_digest: candidate_binding},
        baseline_session_plan=baseline_session_plan,
    )


def prepare_cohort_runtime(
    cohort: CohortPlan,
    *,
    catalog: TargetCatalog,
    expected_context: EvaluationStackContext,
    incumbent_launch: EngineLaunchSpec,
    incumbent_binding: MaterializedArmBinding,
    candidate_bindings: Mapping[str, MaterializedArmBinding],
    baseline_session_plan: SessionExecutionPlan,
) -> PreparedMarginalRuntime:
    """Reopen a sealed cohort and bind B,C1..Ck,B-prime in execution order."""

    if type(cohort) is not CohortPlan or type(catalog) is not TargetCatalog:
        raise MarginalRuntimeError("cohort or catalog has the wrong type")
    try:
        cohort.reopen(catalog=catalog, expected_context=expected_context)
    except (StackPlanError, ValueError, TypeError) as exc:
        raise MarginalRuntimeError(f"cohort is stale or invalid: {exc}") from None
    return _prepare(
        cohort,
        cohort.execution_arms,
        expected_context=expected_context,
        incumbent_launch=incumbent_launch,
        incumbent_binding=incumbent_binding,
        candidate_bindings=candidate_bindings,
        baseline_session_plan=baseline_session_plan,
    )


def prepare_discovery_runtime(
    arm: DiscoveryArmPlan,
    *,
    expected_context: EvaluationStackContext,
    incumbent_launch: EngineLaunchSpec,
    incumbent_binding: MaterializedArmBinding,
    candidate_binding: MaterializedArmBinding,
    baseline_session_plan: SessionExecutionPlan,
) -> PreparedMarginalRuntime:
    """Bind one resolved discovery arm without admitting it to the catalog."""

    if type(arm) is not DiscoveryArmPlan:
        raise MarginalRuntimeError("discovery arm has the wrong type")
    try:
        arm.incumbent.validate_against(expected_context)
    except (TypeError, ValueError) as exc:
        raise MarginalRuntimeError(
            f"discovery arm is stale for this stack context: {exc}"
        ) from None
    return _prepare(
        arm,
        (arm,),
        expected_context=expected_context,
        incumbent_launch=incumbent_launch,
        incumbent_binding=incumbent_binding,
        candidate_bindings={arm.selected_delta_digest: candidate_binding},
        baseline_session_plan=baseline_session_plan,
    )


def _require_execution(
    execution: object,
    *,
    launch: EngineLaunchSpec,
    binding: TrustedLaunchBinding,
    mount: TrustedArenaModelMountReceipt,
    plan: SessionExecutionPlan,
    seen_sessions: set[str],
    seen_device_launches: set[str],
    seen_request_ids: set[str],
    seen_nonces: set[str],
    seen_runtime_policies: set[str],
) -> EngineExecutionEvidence:
    if type(execution) is not EngineExecutionEvidence:
        raise MarginalRuntimeError("executor returned the wrong evidence type")
    receipt = binding.runtime_preflight_receipt
    if type(receipt) is not RuntimePreflightReceipt:
        raise MarginalRuntimeError("execution binding lacks a typed runtime preflight")
    expected_runtime = runtime_identity_from_preflight(receipt)
    if (
        execution.schema != "optima.oci-engine-execution.v1"
        or execution.launch_digest != launch.digest
        or execution.runtime_identity != expected_runtime
        or execution.runtime_preflight_receipt_sha256 != receipt.sha256
        or execution.arena_model_receipt_digest != mount.digest
        or execution.prebuild.launch_digest != launch.digest
        or execution.prebuild.build_spec_digest != binding.native_build_spec.digest
        or execution.prebuild.publication.build_spec_digest
        != binding.native_build_spec.digest
        or execution.native_publication_digest
        != execution.prebuild.publication.publication_digest
    ):
        raise MarginalRuntimeError("executor evidence differs from launch/build/model identity")
    runtime_policy = _digest(
        execution.resource_policy_digest, field="execution runtime resource policy"
    )
    seen_runtime_policies.add(runtime_policy)
    if len(seen_runtime_policies) != 1:
        raise MarginalRuntimeError("executor changed runtime resource policy between arms")
    session = execution.session
    if (
        session.launch_digest != launch.digest
        or session.preflight != plan.expected_preflight
        or session.warmup_count != plan.warmup_count
        or session.conditioning_count != plan.conditioning_count
        or len(session.batches) != len(plan.prompt_batches)
        or tuple(row.batch_index for row in session.batches)
        != tuple(range(len(plan.prompt_batches)))
    ):
        raise MarginalRuntimeError("session evidence differs from the common workload")
    expected_discovery = plan.expected_discovery_overlay_identity_digest
    activation = session.discovery_activation
    if expected_discovery is None:
        if activation is not None:
            raise MarginalRuntimeError(
                "ordinary session acquired discovery activation evidence"
            )
    elif (
        type(activation) is not DiscoveryActivationReceipt
        or activation.overlay_identity_digest != expected_discovery
        or activation.tp_size != plan.engine_config.tp_size
        or activation.driver_origin.version != plan.expected_preflight.sglang_version
    ):
        raise MarginalRuntimeError(
            "discovery session activation differs from its launch policy"
        )
    if session.session_id in seen_sessions:
        raise MarginalRuntimeError("executor replayed a prior session identity")
    request_ids = tuple(row.request_id for row in session.batches)
    nonces = tuple(row.nonce for row in session.batches)
    if (
        len(set(request_ids)) != len(request_ids)
        or len(set(nonces)) != len(nonces)
        or set(request_ids) & seen_request_ids
        or set(nonces) & seen_nonces
    ):
        raise MarginalRuntimeError("executor replayed request or nonce identity")
    for index, row in enumerate(session.batches):
        expected_tokens = len(plan.prompt_batches[index]) * plan.max_new_tokens
        if (
            row.token_numerator != expected_tokens
            or row.evidence.observed_tokens != expected_tokens
        ):
            raise MarginalRuntimeError("session token evidence differs from the workload")
    device_launches = {row.launch_id for row in execution.device_receipts}
    if len(device_launches) != 1:
        raise MarginalRuntimeError("device receipts do not share one launch identity")
    device_launch = next(iter(device_launches))
    if device_launch in seen_device_launches:
        raise MarginalRuntimeError("executor replayed a prior device launch identity")
    seen_sessions.add(session.session_id)
    seen_device_launches.add(device_launch)
    seen_request_ids.update(request_ids)
    seen_nonces.update(nonces)
    return execution


def run_marginal_lifecycle(
    prepared: PreparedMarginalRuntime,
    *,
    executor: EngineExecutor,
    model_mount: TrustedArenaModelMountReceipt,
    deadline: float,
) -> MarginalLifecycleEvidence:
    """Execute exactly B,C1..Ck,B-prime or raise without a partial receipt."""

    if type(prepared) is not PreparedMarginalRuntime:
        raise MarginalRuntimeError("prepared runtime has the wrong type")
    _validate_prepared_runtime(prepared)
    if type(model_mount) is not TrustedArenaModelMountReceipt:
        raise MarginalRuntimeError("model_mount has the wrong type")
    if (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(float(deadline))
        or deadline <= 0
    ):
        raise MarginalRuntimeError("deadline must be one finite positive absolute value")
    if (
        model_mount.arena_digest != prepared.baseline_launch.arena_digest
        or model_mount.model_revision_digest
        != prepared.baseline_launch.model_revision_digest
        or model_mount.model_manifest_digest
        != prepared.baseline_launch.model_manifest_digest
        or model_mount.model_content_digest
        != prepared.baseline_launch.model_content_digest
    ):
        raise MarginalRuntimeError("model mount differs from the prepared launch identity")

    seen_sessions: set[str] = set()
    seen_device_launches: set[str] = set()
    seen_request_ids: set[str] = set()
    seen_nonces: set[str] = set()
    seen_runtime_policies: set[str] = set()

    def execute(
        launch: EngineLaunchSpec,
        binding: TrustedLaunchBinding,
        plan: SessionExecutionPlan,
    ) -> EngineExecutionEvidence:
        raw = executor.execute(
            launch,
            binding,
            model_mount,
            plan,
            deadline=deadline,
        )
        return _require_execution(
            raw,
            launch=launch,
            binding=binding,
            mount=model_mount,
            plan=plan,
            seen_sessions=seen_sessions,
            seen_device_launches=seen_device_launches,
            seen_request_ids=seen_request_ids,
            seen_nonces=seen_nonces,
            seen_runtime_policies=seen_runtime_policies,
        )

    baseline_before = execute(
        prepared.baseline_launch,
        prepared.incumbent_binding.launch_binding,
        prepared.baseline_session_plan,
    )
    candidates: list[CandidateLifecycleEvidence] = []
    for candidate in prepared.candidates:
        execution = execute(
            candidate.launch,
            candidate.binding.launch_binding,
            candidate.session_plan,
        )
        candidates.append(
            CandidateLifecycleEvidence(
                candidate,
                execution,
            )
        )
    baseline_after = execute(
        prepared.baseline_launch,
        prepared.incumbent_binding.launch_binding,
        prepared.baseline_session_plan,
    )
    return MarginalLifecycleEvidence(
        prepared,
        baseline_before,
        tuple(candidates),
        baseline_after,
    )


__all__ = [
    "CandidateLifecycleEvidence",
    "EngineExecutor",
    "MarginalLifecycleEvidence",
    "MarginalRuntimeError",
    "MaterializedArmBinding",
    "PreparedCandidateRuntime",
    "PreparedMarginalRuntime",
    "prepare_cohort_runtime",
    "prepare_discovery_runtime",
    "prepare_marginal_runtime",
    "run_marginal_lifecycle",
]
