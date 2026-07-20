from __future__ import annotations

from types import SimpleNamespace

import pytest

import optima.eval.qualification_intake as intake
from optima.eval.evidence_store import EvidenceArtifactRef
from optima.eval.oci_backend import OCIBackendError
from optima.eval.marginal_runtime import CandidateArmWorkerError
from optima.eval.oci_outer_session import (
    OuterSessionProcessError,
    OuterSessionWorkerError,
)
from optima.eval.qualification import (
    GraphVariantRequirement,
    GraphVerificationBinding,
    GraphVerificationMemberBinding,
    GraphVerificationRequirement,
    QualificationDecision,
)
from optima.eval.qualification_runner import (
    QualificationRunnerError,
    SpeedStageDisposition,
)
from optima.eval.scoring import RawSpeedEvidenceError
from optima.verify import VerifyResult


def _d(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode()).hexdigest()


def _reservation(index: int, delta: str | None = None) -> intake.QualificationReservation:
    return intake.QualificationReservation(
        _d(f"reservation-{index}"),
        _d(f"submission-{index}"),
        f"target.{index}",
        delta or _d(f"delta-{index}"),
        index,
        f"miner-{index}",
        100 + index,
        index,
        0,
        (f"target.{index}",),
    )


def _fake_plan(monkeypatch, *, count: int = 2, discovery: bool = False):
    class FakePlan:
        pass

    class FakeDiscovery:
        pass

    monkeypatch.setattr(intake, "CausalQualificationInput", FakePlan)
    if discovery:
        monkeypatch.setattr(intake, "DiscoveryArmPlan", FakeDiscovery)
        source = FakeDiscovery()
    else:
        source = SimpleNamespace()
    source.digest = _d("source")
    plan = FakePlan()
    plan.selection_secret = b"s" * 32
    plan.prepared = SimpleNamespace(source=source)
    plan.commitment = SimpleNamespace(digest=_d("commitment"))
    plan.candidates = tuple(
        SimpleNamespace(selected_delta_digest=_d(f"delta-{index}"))
        for index in range(count)
    )
    plan.evidence_root = SimpleNamespace()
    plan.speed_stage_disposition = SpeedStageDisposition.TERMINAL
    monkeypatch.setattr(
        intake, "qualification_authority_digest", lambda _value: _d("authority")
    )
    reservations = tuple(_reservation(index) for index in range(count))
    manifest = intake.QualificationAuthorityManifest.seal(
        plan,
        reservations=reservations,
        selection_secret_reference=_d("secret-reference"),
    )
    return plan, manifest


def _factory(plan, manifest):
    return intake.QualificationPlanFactory(
        manifest,
        lambda reference: (
            plan.selection_secret
            if reference == manifest.selection_secret_reference
            else b""
        ),
        lambda secret: plan,
    )


def _requirement() -> GraphVerificationRequirement:
    slot, variant = "collective.all_reduce", "default"
    descriptors = tuple(sorted((_d("shape-a"), _d("shape-b"))))
    member = GraphVerificationMemberBinding(
        slot, _d("target-spec"), _d("contract"), "collective.tp8"
    )
    binding = GraphVerificationBinding(
        _d("arm"),
        _d("launch"),
        _d("contribution"),
        _d("delta"),
        slot,
        _d("target-spec"),
        _d("catalog"),
        (member,),
        _d("verification-policy"),
    )
    return GraphVerificationRequirement(
        binding,
        (
            GraphVariantRequirement(
                slot, variant, descriptors, True, descriptors
            ),
        ),
        3,
    )


def _graph_observation(
    requirement: GraphVerificationRequirement,
) -> intake.GraphVerificationObservation:
    variant = requirement.variants[0]
    shapes = tuple(
        intake.GraphShapeObservation(
            descriptor, True, True, True, requirement.expected_graph_replays, True
        )
        for descriptor in variant.shape_descriptor_digests
    )
    return intake.GraphVerificationObservation(
        requirement.digest,
        (
            intake.GraphMemberObservation(
                variant.slot_id,
                (
                    intake.GraphVariantObservation(
                        variant.slot_id,
                        variant.variant_id,
                        True,
                        True,
                        shapes,
                    ),
                ),
            ),
        ),
    )


def test_authority_manifest_roundtrip_contains_only_private_secret_reference(
    monkeypatch,
) -> None:
    plan, manifest = _fake_plan(monkeypatch)

    encoded = manifest.to_dict()
    assert intake.QualificationAuthorityManifest.from_dict(encoded) == manifest
    assert manifest.digest == intake.QualificationAuthorityManifest.from_dict(
        encoded
    ).digest
    assert plan.selection_secret.hex() not in str(encoded)
    assert encoded["selection_secret_reference"] == _d("secret-reference")


def test_plan_factory_reopens_exact_secret_and_public_authority(monkeypatch) -> None:
    plan, manifest = _fake_plan(monkeypatch)
    factory = _factory(plan, manifest)

    assert factory.build() is plan

    substituted = intake.QualificationPlanFactory(
        manifest, lambda _reference: b"x" * 32, lambda _secret: plan
    )
    with pytest.raises(intake.QualificationIntakeError, match="substituted"):
        substituted.build()

    changed = SimpleNamespace(**plan.__dict__)
    changed.selection_secret = plan.selection_secret
    with pytest.raises(intake.QualificationIntakeError, match="untyped plan"):
        intake.QualificationPlanFactory(
            manifest, lambda _reference: plan.selection_secret, lambda _secret: changed
        ).build()


def test_graph_observation_publishes_and_reopens_canonical_raw_facts(tmp_path) -> None:
    requirement = _requirement()
    product = intake.publish_graph_observation(
        tmp_path / "evidence", requirement, _graph_observation(requirement)
    )

    assert product.requirement_digest == requirement.digest
    assert product.evidence_ref.raw_evidence_digest == product.raw_evidence_digest
    assert product.grade.decision is QualificationDecision.PASS
    assert product.grade.reason == "graph_verification_pass"


def test_graph_regrade_enforces_the_required_replay_count(tmp_path) -> None:
    requirement = _requirement()
    observation = _graph_observation(requirement)
    variant = observation.members[0].variants[0]
    short = tuple(
        intake.GraphShapeObservation(
            row.descriptor_digest, True, True, True, 2, True
        )
        for row in variant.shapes
    )
    altered = intake.GraphVerificationObservation(
        requirement.digest,
        (
            intake.GraphMemberObservation(
                variant.slot_id,
                (
                    intake.GraphVariantObservation(
                        variant.slot_id, variant.variant_id, True, True, short
                    ),
                ),
            ),
        ),
    )

    product = intake.publish_graph_observation(
        tmp_path / "evidence", requirement, altered
    )
    assert product.grade.decision is QualificationDecision.NO_DECISION
    assert product.grade.reason == "graph_replay_count_mismatch"


@pytest.mark.parametrize(
    "aggregate",
    [True, VerifyResult("collective.all_reduce", "bfloat16", True, [])],
)
def test_graph_adapter_rejects_aggregate_verdicts(tmp_path, aggregate) -> None:
    with pytest.raises(
        intake.QualificationIntakeError, match="not VerifyResult or booleans"
    ):
        intake.publish_graph_observation(
            tmp_path / "evidence", _requirement(), aggregate
        )


def test_graph_observation_requires_causal_eager_capture_replay_facts() -> None:
    with pytest.raises(intake.QualificationIntakeError, match="causally inconsistent"):
        intake.GraphShapeObservation(_d("shape"), True, False, True, 1, True)
    with pytest.raises(intake.QualificationIntakeError, match="exact boolean"):
        intake.GraphShapeObservation(_d("shape"), 1, True, True, 3, True)  # type: ignore[arg-type]


def test_graph_adapter_rejects_missing_descriptor_even_with_passing_facts(
    tmp_path,
) -> None:
    requirement = _requirement()
    observation = _graph_observation(requirement)
    variant = observation.members[0].variants[0]
    missing = intake.GraphVerificationObservation(
        requirement.digest,
        (
            intake.GraphMemberObservation(
                variant.slot_id,
                (
                    intake.GraphVariantObservation(
                        variant.slot_id,
                        variant.variant_id,
                        True,
                        True,
                        variant.shapes[:-1],
                    ),
                ),
            ),
        ),
    )
    with pytest.raises(intake.QualificationIntakeError, match="shape observations"):
        intake.publish_graph_observation(
            tmp_path / "evidence", requirement, missing
        )


class _FakeReport:
    def __init__(self, delta: str, decision: QualificationDecision, index: int):
        self.selected_delta_digest = delta
        self.decision = decision
        self.reason = {
            QualificationDecision.PASS: "qualified",
            QualificationDecision.FAIL: "speed_regression",
            QualificationDecision.NO_DECISION: "speed_noise",
        }[decision]
        self.retryable = decision is QualificationDecision.NO_DECISION
        self.digest = _d(f"report-{index}")


class _FakeAttempt:
    pass


class _FakeStageExit:
    def __init__(
        self,
        manifest: intake.QualificationAuthorityManifest,
        decision: QualificationDecision,
    ) -> None:
        self.authority_digest = manifest.authority_digest
        self.source_digest = manifest.source_digest
        self.selected_delta_digest = manifest.reservations[0].selected_delta_digest
        self.stage = "speed"
        self.decision = decision
        self.reason = (
            "speed_noise"
            if decision is QualificationDecision.NO_DECISION
            else "speed_regression"
        )
        self.digest = _d(f"stage-exit-{decision.value}")


def _install_success_runner(monkeypatch, manifest, decisions):
    reference = EvidenceArtifactRef(
        "qualification.cohort-attempt",
        _d("attempt-artifact"),
        1,
        "application/json",
        "optima.qualification.cohort-attempt.v1",
    )
    reports = tuple(
        _FakeReport(delta, decision, index)
        for index, (delta, decision) in enumerate(
            zip(manifest.candidate_deltas, decisions, strict=True)
        )
    )
    attempt = _FakeAttempt()
    attempt.authority_digest = manifest.authority_digest
    attempt.source_digest = manifest.source_digest
    attempt.reports = reports
    monkeypatch.setattr(intake, "CohortQualificationAttempt", _FakeAttempt)
    monkeypatch.setattr(intake, "CandidateQualificationReport", _FakeReport)
    monkeypatch.setattr(
        intake, "run_causal_qualification", lambda *_args, **_kwargs: reference
    )
    monkeypatch.setattr(
        intake, "reopen_causal_qualification", lambda *_args, **_kwargs: attempt
    )
    return reference


@pytest.mark.parametrize(
    ("decision", "reason", "expects_retry"),
    (
        (QualificationDecision.FAIL, "speed_regression", False),
        (QualificationDecision.NO_DECISION, "speed_noise", True),
    ),
)
def test_speed_stage_exit_projects_terminal_outcome_without_settlement(
    monkeypatch,
    decision: QualificationDecision,
    reason: str,
    expects_retry: bool,
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=1)
    reference = EvidenceArtifactRef(
        "qualification.stage-exit",
        _d(f"stage-exit-artifact-{decision.value}"),
        1,
        "application/json",
        intake.STAGE_EXIT_SCHEMA,
    )
    terminal = _FakeStageExit(manifest, decision)
    resident_baseline_executor = object()
    runner_kwargs = {}

    def run(*_args, **kwargs):
        runner_kwargs.update(kwargs)
        return reference

    monkeypatch.setattr(intake, "QualificationStageExit", _FakeStageExit)
    monkeypatch.setattr(intake, "run_causal_qualification", run)
    monkeypatch.setattr(
        intake,
        "reopen_qualification_stage_exit",
        lambda *_args, **_kwargs: terminal,
    )
    monkeypatch.setattr(
        intake,
        "reopen_causal_qualification",
        lambda *_args, **_kwargs: pytest.fail(
            "a terminal stage exit must not reopen a full qualification attempt"
        ),
    )

    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        resident_baseline_executor=resident_baseline_executor,
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert runner_kwargs["resident_baseline_executor"] is resident_baseline_executor
    assert result.attempt_ref == reference
    assert len(result.outcomes) == 1
    outcome = result.outcomes[0]
    assert outcome.decision is decision
    assert outcome.reason == reason
    assert outcome.retryable is expects_retry
    assert outcome.attempt_artifact_sha256 == reference.sha256
    assert outcome.report_digest == terminal.digest
    assert outcome.failure_digest is None
    assert outcome.settlement_qualification is None
    if expects_retry:
        assert result.retry_plan is not None
        assert result.retry_plan.strategy == "requeue"
        assert result.retry_plan.reservation_groups == (
            (manifest.reservations[0].reservation_digest,),
        )
    else:
        assert result.retry_plan is None


def test_intake_rejects_calibration_observation_before_runner(
    monkeypatch,
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=1)
    plan.speed_stage_disposition = SpeedStageDisposition.CALIBRATION_OBSERVATION
    monkeypatch.setattr(
        intake,
        "run_causal_qualification",
        lambda *_args, **_kwargs: pytest.fail(
            "economic intake must reject calibration authority before the runner"
        ),
    )

    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        resident_baseline_executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert result.attempt_ref is None
    assert len(result.outcomes) == 1
    outcome = result.outcomes[0]
    assert outcome.decision is QualificationDecision.NO_DECISION
    assert outcome.reason == "qualification_plan"
    assert outcome.retryable is True
    assert outcome.report_digest is None
    assert outcome.failure_digest is not None
    assert outcome.settlement_qualification is None


def test_batch_service_projects_per_reservation_tristate_and_retry(monkeypatch) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=3)
    reference = _install_success_runner(
        monkeypatch,
        manifest,
        (
            QualificationDecision.PASS,
            QualificationDecision.FAIL,
            QualificationDecision.NO_DECISION,
        ),
    )

    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert [row.decision for row in result.outcomes] == [
        QualificationDecision.PASS,
        QualificationDecision.FAIL,
        QualificationDecision.NO_DECISION,
    ]
    assert all(row.attempt_artifact_sha256 == reference.sha256 for row in result.outcomes)
    assert result.retry_plan is not None
    assert result.retry_plan.strategy == "requeue"
    assert result.retry_plan.reservation_groups == (
        (manifest.reservations[2].reservation_digest,),
    )


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (RawSpeedEvidenceError("zero throughput"), "raw_speed_evidence"),
        (QualificationRunnerError("T died"), "qualification_runner"),
        (
            OuterSessionProcessError("session ended before a complete response"),
            "outer_session_process",
        ),
        (OCIBackendError("runtime post-drain unavailable"), "oci_backend"),
    ],
)
def test_cohort_failure_is_no_decision_with_deterministic_bisection(
    monkeypatch, failure, reason
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=3)

    def fail(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(intake, "run_causal_qualification", fail)
    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert {row.decision for row in result.outcomes} == {
        QualificationDecision.NO_DECISION
    }
    assert all(row.retryable and row.report_digest is None for row in result.outcomes)
    assert all(row.reason == reason for row in result.outcomes)
    assert result.attempt_ref is None
    assert result.retry_plan is not None
    assert result.retry_plan.strategy == "bisect"
    assert result.retry_plan.reservation_groups == (
        (manifest.reservations[0].reservation_digest,),
        tuple(row.reservation_digest for row in manifest.reservations[1:]),
    )


def test_candidate_worker_error_is_contained_by_deterministic_bisection(
    monkeypatch,
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=3)
    failure = CandidateArmWorkerError(
        candidate_index=1,
        selected_delta_digest=manifest.reservations[1].selected_delta_digest,
        arm_digest=_d("candidate-arm"),
        launch_digest=_d("candidate-launch"),
        worker_error=OuterSessionWorkerError("candidate engine raised"),
    )
    monkeypatch.setattr(
        intake,
        "run_causal_qualification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert all(
        row.decision is QualificationDecision.NO_DECISION
        and row.retryable
        and row.reason == "candidate_worker"
        and row.report_digest is None
        and row.failure_digest is not None
        for row in result.outcomes
    )
    assert result.attempt_ref is None
    assert result.retry_plan is not None
    assert result.retry_plan.strategy == "bisect"
    assert result.retry_plan.reservation_groups == (
        (manifest.reservations[0].reservation_digest,),
        tuple(row.reservation_digest for row in manifest.reservations[1:]),
    )


@pytest.mark.parametrize("source", ("baseline", "reference"))
def test_shared_worker_error_is_not_attributed_to_candidate(
    monkeypatch, source: str
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=2)
    monkeypatch.setattr(
        intake,
        "run_causal_qualification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OuterSessionWorkerError(f"{source} worker raised")
        ),
    )

    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert all(
        row.decision is QualificationDecision.NO_DECISION
        and row.reason == "outer_session_worker"
        and row.reason != "candidate_worker"
        for row in result.outcomes
    )
    assert result.retry_plan is not None
    assert result.retry_plan.strategy == "bisect"


def test_candidate_worker_identity_mismatch_is_a_controller_failure(
    monkeypatch,
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=2)
    failure = CandidateArmWorkerError(
        candidate_index=1,
        selected_delta_digest=manifest.reservations[0].selected_delta_digest,
        arm_digest=_d("candidate-arm"),
        launch_digest=_d("candidate-launch"),
        worker_error=OuterSessionWorkerError("candidate engine raised"),
    )
    monkeypatch.setattr(
        intake,
        "run_causal_qualification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )

    with pytest.raises(
        intake.QualificationIntakeError, match="candidate worker identity"
    ):
        intake.run_qualification_intake(
            _factory(plan, manifest),
            executor=object(),
            entropy_provider=lambda *_args: None,
            hidden_judge=lambda **_kwargs: None,
            deadline=100.0,
        )


def test_unexpected_controller_failure_still_propagates(monkeypatch) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=2)
    monkeypatch.setattr(
        intake,
        "run_causal_qualification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("controller invariant failed")
        ),
    )

    with pytest.raises(RuntimeError, match="controller invariant failed"):
        intake.run_qualification_intake(
            _factory(plan, manifest),
            executor=object(),
            entropy_provider=lambda *_args: None,
            hidden_judge=lambda **_kwargs: None,
            deadline=100.0,
        )


@pytest.mark.parametrize(
    "failure",
    [
        intake.QualificationIntakeError("secret record unavailable"),
        OSError("private store unavailable"),
    ],
)
def test_factory_infrastructure_failure_is_typed_no_decision(
    monkeypatch, failure
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=2)

    def fail_secret(_reference):
        raise failure

    factory = intake.QualificationPlanFactory(manifest, fail_secret, lambda _secret: plan)
    result = intake.run_qualification_intake(
        factory,
        executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )

    assert all(
        row.decision is QualificationDecision.NO_DECISION
        and row.reason == "qualification_plan"
        for row in result.outcomes
    )
    assert result.retry_plan is not None
    assert result.retry_plan.strategy == "bisect"


def test_discovery_is_singleton_and_shared_failure_requeues_without_bisection(
    monkeypatch,
) -> None:
    plan, manifest = _fake_plan(monkeypatch, count=1, discovery=True)
    monkeypatch.setattr(
        intake,
        "run_causal_qualification",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            QualificationRunnerError("discovery T failed")
        ),
    )

    result = intake.run_qualification_intake(
        _factory(plan, manifest),
        executor=object(),
        entropy_provider=lambda *_args: None,
        hidden_judge=lambda **_kwargs: None,
        deadline=100.0,
    )
    assert manifest.lane == "discovery"
    assert result.retry_plan is not None
    assert result.retry_plan.strategy == "requeue"
    assert result.retry_plan.reservation_groups == (
        (manifest.reservations[0].reservation_digest,),
    )

    with pytest.raises(intake.QualificationIntakeError, match="reservations"):
        intake.QualificationAuthorityManifest(
            "discovery",
            manifest.authority_digest,
            manifest.source_digest,
            manifest.commitment_digest,
            manifest.selection_secret_reference,
            (_d("delta-a"), _d("delta-b")),
            (_reservation(0, _d("delta-a")), _reservation(1, _d("delta-b"))),
        )


def test_outcomes_and_batches_cannot_claim_evidence_free_pass() -> None:
    with pytest.raises(intake.QualificationIntakeError, match="PASS/FAIL"):
        intake.QualificationIntakeOutcome(
            _d("reservation"),
            _d("delta"),
            _d("authority"),
            QualificationDecision.PASS,
            "qualified",
            False,
        )

    reference = EvidenceArtifactRef(
        "qualification.cohort-attempt",
        _d("attempt"),
        1,
        "application/json",
        "optima.qualification.cohort-attempt.v1",
    )
    outcome = intake.QualificationIntakeOutcome(
        _d("reservation"),
        _d("delta"),
        _d("authority"),
        QualificationDecision.FAIL,
        "quality_failed",
        False,
        attempt_artifact_sha256=reference.sha256,
        report_digest=_d("report"),
    )
    with pytest.raises(intake.QualificationIntakeError, match="internally inconsistent"):
        intake.QualificationIntakeBatch(_d("authority"), (outcome,))


def test_single_pass_outcome_cannot_smuggle_a_settlement_candidate() -> None:
    reference = EvidenceArtifactRef(
        "qualification.cohort-attempt",
        _d("attempt"),
        1,
        "application/json",
        "optima.qualification.cohort-attempt.v1",
    )
    with pytest.raises(
        intake.QualificationIntakeError, match="settlement qualification"
    ):
        intake.QualificationIntakeOutcome(
            _d("reservation"),
            _d("delta"),
            _d("authority"),
            QualificationDecision.PASS,
            "qualified",
            False,
            attempt_artifact_sha256=reference.sha256,
            report_digest=_d("report"),
            settlement_qualification=object(),  # type: ignore[arg-type]
        )
