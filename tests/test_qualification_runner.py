from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import optima.eval.qualification_runner as runner
from optima.eval.device_state import DeviceStateReceipt, DeviceStateSample
from optima.eval.evidence_store import EvidenceArtifactRef, publish_evidence, reopen_evidence
from optima.eval.oci_backend import OCIBackendError, OCIEngineExecutor
from optima.eval.oci_process import OCIQuiescenceReceipt
from optima.eval.qualification import (
    DiscoveryQualificationProfile,
    GraphVerificationGrade,
    QualificationDecision,
    QualificationError,
    SelectionCommitment,
    SelectionEntropyReceipt,
)
from optima.eval.reference_protocol import ReferenceRoleInput, ReferenceTokenEvidence
from optima.eval.reference_quality import ReferenceQualityVerdict
from tests.test_qualification import _discovery_execution, _reference


_REAL_PUBLISH_CAUSAL = runner.publish_causal_qualification
_REAL_REOPEN_CAUSAL = runner.reopen_causal_qualification
_REAL_QUALIFICATION_AUTHORITY = runner.qualification_authority_digest


def _d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _artifact(label: str) -> EvidenceArtifactRef:
    return EvidenceArtifactRef(
        "reference-quality.raw",
        _d(label),
        2,
        "application/json",
        "optima.reference-quality-raw.v1",
    )


def _quiescence(sequence: int, observed: float) -> OCIQuiescenceReceipt:
    return OCIQuiescenceReceipt(
        "optima.oci-quiescence.v1",
        "runner",
        "1" * 32,
        _d("executor-namespace"),
        sequence,
        observed,
        (),
        (),
        (),
    )


def _graph_grade(decision: QualificationDecision, index: int) -> GraphVerificationGrade:
    reason = {
        QualificationDecision.PASS: "graph_verification_pass",
        QualificationDecision.FAIL: "graph_replay_failed",
        QualificationDecision.NO_DECISION: "graph_evidence_missing",
    }[decision]
    return GraphVerificationGrade(
        decision,
        reason,
        _d(f"graph-requirement-{index}"),
        _d(f"graph-reference-{index}"),
        _d(f"graph-raw-{index}"),
    )


def _quality_verdict(
    decision: QualificationDecision, index: int, calibration_digest: str
) -> ReferenceQualityVerdict:
    failed = ("mean_nll",) if decision is QualificationDecision.FAIL else ()
    overlap = ("mean_nll",) if decision is QualificationDecision.NO_DECISION else ()
    return ReferenceQualityVerdict(
        decision.value,
        failed,
        overlap,
        "1",
        _d(f"quality-evidence-{index}"),
        calibration_digest,
    )


class _Harness:
    def __init__(
        self,
        monkeypatch,
        *,
        graph: tuple[QualificationDecision, ...],
        speed: tuple[QualificationDecision, ...],
        quality: tuple[QualificationDecision, ...],
        swap_exchanges: bool = False,
        fail_pre_t_quiescence: bool = False,
        exercise_judge_cache: bool = False,
    ) -> None:
        assert len(graph) == len(speed) == len(quality)
        # This runner harness predates the typed authority constructor and
        # intentionally replaces the fully validated input boundary with
        # lightweight records. Preserve exact production type checks while
        # identifying only these registered-lane test doubles as that type.
        monkeypatch.setattr(
            runner, "CandidateQualificationAuthority", SimpleNamespace
        )
        self.calls: list[str] = []
        self.reference_calls = 0
        self.reference_request_counts: list[int] = []
        self.secret = b"s" * 32
        secret_commitment = hashlib.sha256(
            b"optima-selection-secret-v1\0" + self.secret
        ).hexdigest()
        self.commitment = SelectionCommitment(
            _d("source"),
            _d("reference"),
            _d("workload"),
            _d("entropy-source"),
            secret_commitment,
            (_d("prompt"),),
            1,
        )
        self.entropy = SelectionEntropyReceipt(
            self.commitment.entropy_source_digest,
            self.commitment.digest,
            _d("entropy"),
            _d("entropy-authority"),
        )
        self.calibration = SimpleNamespace(digest=_d("calibration"))
        self.grades = {
            _d(f"delta-{index}"): _graph_grade(decision, index)
            for index, decision in enumerate(graph)
        }
        reference = SimpleNamespace(
            digest=self.commitment.reference_manifest_digest,
            pristine_launch_digest=_d("pristine-launch"),
            hidden_corpus_commitment=_d("hidden-corpus"),
            hidden_judge_digest=_d("hidden-judge"),
        )
        hidden_task_policy = _d("hidden-task-policy")
        authorities = tuple(
            SimpleNamespace(
                selected_delta_digest=_d(f"delta-{index}"),
                profile=SimpleNamespace(
                    reference=reference,
                    digest=_d(f"profile-{index}"),
                    hidden_task_policy_digest=hidden_task_policy,
                ),
                graph_requirement=SimpleNamespace(digest=_d(f"requirement-{index}")),
                graph_artifact_ref=_artifact(f"graph-{index}"),
                graph_evidence_ref=SimpleNamespace(
                    digest=_d(f"graph-evidence-ref-{index}")
                ),
            )
            for index in range(len(graph))
        )
        lifecycle_candidates = tuple(
            SimpleNamespace(
                arm=SimpleNamespace(
                    selected_delta_digest=authority.selected_delta_digest,
                    digest=_d(f"arm-{index}"),
                    transition=SimpleNamespace(target_id=f"target.{index}"),
                ),
                candidate=SimpleNamespace(
                    launch=SimpleNamespace(digest=_d(f"candidate-launch-{index}"))
                ),
            )
            for index, authority in enumerate(authorities)
        )
        self.lifecycle = SimpleNamespace(
            candidates=lifecycle_candidates,
            baseline_after=SimpleNamespace(
                device_receipts=(
                    SimpleNamespace(completed_monotonic_s=1.0),
                    SimpleNamespace(completed_monotonic_s=2.0),
                )
            ),
        )
        prepared = SimpleNamespace(
            source=SimpleNamespace(digest=_d("source")),
            candidates=tuple(
                SimpleNamespace(
                    arm=row.arm,
                    launch=row.candidate.launch,
                    binding=SimpleNamespace(
                        launch_binding=SimpleNamespace(
                            native_build_spec=SimpleNamespace(
                                digest=_d(f"candidate-native-{index}")
                            ),
                            runtime_preflight_receipt=SimpleNamespace(
                                sha256=_d(f"candidate-preflight-{index}")
                            ),
                        )
                    ),
                )
                for index, row in enumerate(lifecycle_candidates)
            ),
            baseline_launch=SimpleNamespace(digest=_d("baseline-launch")),
            incumbent_binding=SimpleNamespace(
                launch_binding=SimpleNamespace(
                    runtime_preflight_receipt=SimpleNamespace(
                        sha256=_d("incumbent-preflight")
                    )
                )
            ),
        )
        self.value = SimpleNamespace(
            prepared=prepared,
            model_mount=SimpleNamespace(digest=_d("model-mount")),
            candidates=authorities,
            commitment=self.commitment,
            selection_secret=self.secret,
            evidence_root=SimpleNamespace(),
            pristine_stack=SimpleNamespace(digest=_d("pristine-stack")),
            pristine_launch=SimpleNamespace(digest=_d("pristine-launch")),
            pristine_binding=SimpleNamespace(
                native_build_spec=SimpleNamespace(digest=_d("pristine-native")),
                controller_distribution_digest=_d("pristine-controller"),
                runtime_preflight_receipt=SimpleNamespace(
                    sha256=_d("pristine-preflight")
                ),
            ),
            reference_engine_config=SimpleNamespace(digest=_d("reference-engine-config")),
            reference_preflight=SimpleNamespace(digest=_d("reference-preflight")),
            expected_launch_resource_policy_digest=_d("launch-policy"),
            expected_runtime_resource_policy_digest=_d("runtime-policy"),
            expected_device_policy_digest=_d("device-policy"),
            calibration_threshold_policy=SimpleNamespace(
                digest=_d("calibration-threshold-policy")
            ),
            calibration_manifest=self.calibration,
            calibration_artifact_ref=_artifact("calibration"),
            calibration_context=SimpleNamespace(
                digest=_d("calibration-context"),
                workload_digest=_d("workload"),
            ),
        )

        self.executor = object.__new__(OCIEngineExecutor)
        self.executor.manager = SimpleNamespace(clock=lambda: 3.5)
        before, after = _quiescence(1, 3.0), _quiescence(2, 6.0)
        quiescence_count = 0

        @contextmanager
        def transaction(_executor):
            self.calls.append("transaction.enter")
            try:
                yield _executor
            finally:
                self.calls.append("transaction.exit")

        def prove_quiescent(_executor):
            nonlocal quiescence_count
            quiescence_count += 1
            self.calls.append(f"quiescence.{quiescence_count}")
            if fail_pre_t_quiescence and quiescence_count == 1:
                raise runner.QualificationRunnerError("pre-T quiescence failed")
            return before if quiescence_count == 1 else after

        def execute_reference(_executor, _launch, _binding, _mount, plan, *, deadline):
            del deadline
            self.calls.append("reference")
            self.reference_calls += 1
            self.reference_request_counts.append(len(plan.requests))
            requests = tuple(plan.requests)
            if swap_exchanges:
                requests = tuple(reversed(requests))
            exchanges = tuple(
                SimpleNamespace(
                    request=request,
                    request_sha256=request.sha256,
                    evidence_frame_sha256=_d(f"evidence-{request.index}"),
                )
                for request in requests
            )
            session = SimpleNamespace(
                exchanges=exchanges,
                digest=_d("reference-session"),
                session_id=requests[0].session_id,
                request_plan_digest=requests[0].plan_digest,
            )
            sample = DeviceStateSample(4.0, (), (), True, "idle")
            receipts = (
                DeviceStateReceipt(
                    "optima.device-state-receipt.v1", 10, "reference-launch", "pre",
                    (0,), _d("device-config"), self.value.expected_device_policy_digest,
                    4.0, 4.1, 1, (sample,),
                ),
                DeviceStateReceipt(
                    "optima.device-state-receipt.v1", 11, "reference-launch", "post",
                    (0,), _d("device-config"), self.value.expected_device_policy_digest,
                    4.2, 5.0, 1, (sample,),
                ),
            )
            return SimpleNamespace(
                launch_digest=reference.pristine_launch_digest,
                runtime_identity=SimpleNamespace(
                    runtime_digest=_d("reference-runtime"),
                    base_engine_digest=_d("reference-base"),
                    validator_overlay_digest=_d("reference-overlay"),
                ),
                runtime_preflight_receipt_sha256=_d("reference-preflight-receipt"),
                arena_model_receipt_digest=_d("model-mount"),
                resource_policy_digest=self.value.expected_runtime_resource_policy_digest,
                prebuild=SimpleNamespace(build_spec_digest=_d("reference-build-spec")),
                native_publication_digest=_d("reference-publication"),
                runtime_argv_sha256=_d("reference-argv"),
                recovered_lease_ids=(),
                session=session,
                device_receipts=receipts,
            )

        monkeypatch.setattr(OCIEngineExecutor, "exclusive_transaction", transaction)
        monkeypatch.setattr(OCIEngineExecutor, "prove_quiescent", prove_quiescent)
        monkeypatch.setattr(OCIEngineExecutor, "execute_reference", execute_reference)

        def prevalidate(_value, *_args, **_kwargs):
            self.calls.append("prevalidate")
            return self.calibration, self.grades

        monkeypatch.setattr(runner, "_validate_pre_execution", prevalidate)
        monkeypatch.setattr(
            runner,
            "run_marginal_lifecycle",
            lambda *_args, **_kwargs: self.calls.append("lifecycle") or self.lifecycle,
        )
        monkeypatch.setattr(runner, "cohort_trajectory_digest", lambda _row: _d("cohort"))

        def entropy_provider(commitment, teardown):
            assert commitment is self.commitment
            assert teardown is before
            self.calls.append("entropy")
            return self.entropy

        self.entropy_provider = entropy_provider

        class BoundJudge:
            binding = runner.HiddenJudgeBinding(
                reference.hidden_corpus_commitment,
                reference.hidden_judge_digest,
                hidden_task_policy,
            )

            def __init__(self):
                self.calls = 0

            def __call__(self, *, prompt_digest, output_ids, task_digests):
                self.calls += 1
                return runner.HiddenJudgeReceipt(
                    self.binding.digest,
                    prompt_digest,
                    runner.hidden_judge_output_digest(prompt_digest, output_ids),
                    task_digests,
                    (True,) * len(task_digests),
                )

        self.hidden_judge = BoundJudge()

        def make_request(
            _lifecycle,
            authority,
            _selection,
            *,
            session_id,
            plan_digest,
            request_id,
            nonce,
            index,
        ):
            del request_id, nonce
            return SimpleNamespace(
                index=index,
                delta=authority.selected_delta_digest,
                session_id=session_id,
                plan_digest=plan_digest,
                sha256=_d(f"request-{authority.selected_delta_digest}"),
            )

        monkeypatch.setattr(runner, "_reference_request", make_request)
        monkeypatch.setattr(runner, "request_sha256", lambda request: request.sha256)
        monkeypatch.setattr(
            runner,
            "ReferenceSessionPlan",
            lambda *_args: SimpleNamespace(digest=_d("reference-plan"), requests=_args[-1]),
        )
        raw_index = 0

        def raw_artifact(_lifecycle, authority, *_args):
            nonlocal raw_index
            self.calls.append(f"raw.{raw_index}")
            raw_index += 1
            if exercise_judge_cache:
                judge = _args[-1]
                for _ in range(3):
                    judge(
                        prompt_digest=_d("memo-prompt"),
                        output_ids=(1, 2),
                        task_digests=(_d("memo-task"),),
                    )
            binding = object.__new__(runner.ReferenceQualityRawBinding)
            binding_values = {
                "qualification_identity_digest": _d("qualification-identity"),
                "reference_manifest_digest": self.commitment.reference_manifest_digest,
                "calibration_digest": self.calibration.digest,
                "selection_digest": _d("selection"),
                "candidate_lifecycle_digest": _d("candidate-lifecycle"),
                "selected_trajectory_digest": _d("selected-trajectory"),
                "selected_trajectory_projection_digest": _d("trajectory-projection"),
                "selected_prompt_digests": (_d("prompt"),),
                "t_session_digest": _d("reference-session"),
                "t_request_sha256": _d(f"request-{authority.selected_delta_digest}"),
                "support_policy_digest": _d("support-policy"),
                "hidden_task_plan_digest": _d("hidden-task-plan"),
                "nll_tail_threshold": "1",
                "tokens_per_prompt": 1,
                "topk_width": 1,
                "hidden_tasks_per_prompt": 1,
            }
            for name, value in binding_values.items():
                object.__setattr__(binding, name, value)
            return SimpleNamespace(
                binding=binding,
                to_dict=lambda: {"candidate": authority.selected_delta_digest},
            )

        monkeypatch.setattr(runner, "_raw_artifact", raw_artifact)
        monkeypatch.setattr(
            runner,
            "validate_quality_binding",
            lambda *_args, **_kwargs: self.calls.append("quality.validate"),
        )
        publish_index = 0

        def publish(*_args, **_kwargs):
            nonlocal publish_index
            self.calls.append("quality.publish")
            result = _artifact(f"raw-quality-{publish_index}")
            publish_index += 1
            return result

        monkeypatch.setattr(runner, "publish_evidence", publish)
        monkeypatch.setattr(
            runner,
            "reopen_reference_quality_evidence",
            lambda *_args, **_kwargs: self.calls.append("quality.reopen") or object(),
        )
        quality_rows = iter(enumerate(quality))

        def score_quality(*_args, **_kwargs):
            index, decision = next(quality_rows)
            self.calls.append(f"quality.score.{index}")
            return _quality_verdict(decision, index, self.calibration.digest)

        monkeypatch.setattr(runner, "score_reference_quality", score_quality)
        speed_rows = iter(enumerate(speed))

        def project_speed(*_args, **_kwargs):
            index, decision = next(speed_rows)
            self.calls.append(f"speed.{index}")
            baseline_launch = _d("baseline-launch")
            candidate_launch = _d(f"candidate-launch-{index}")

            def rate(label, launch, tokens):
                return runner.ChargedExecutionRate(
                    launch,
                    f"{100 + index * 10 + label:032x}",
                    tokens // 3,
                    tokens - tokens // 3,
                    tokens,
                    1.0,
                    2.0,
                    3.0,
                    tokens / 3.0,
                )

            before = rate(1, baseline_launch, 30)
            candidate = rate(
                2,
                candidate_launch,
                36 if decision is QualificationDecision.PASS else 27
                if decision is QualificationDecision.FAIL else 30,
            )
            after = rate(
                3,
                baseline_launch,
                60 if decision is QualificationDecision.NO_DECISION else 30,
            )
            context_digest = _d("calibration-context")
            workload_digest = _d("workload")
            evidence_digest = runner._projection_digest(
                _d(f"delta-{index}"),
                candidate_launch,
                self.calibration.digest,
                context_digest,
                workload_digest,
                self.value.expected_runtime_resource_policy_digest,
                (before, candidate, after),
            )
            verdict = runner.score_speedup(
                [before.tokens_per_second, after.tokens_per_second],
                candidate.tokens_per_second,
            )
            return runner.MarginalSpeedProjection(
                _d(f"delta-{index}"),
                candidate_launch,
                self.calibration.digest,
                context_digest,
                workload_digest,
                evidence_digest,
                before,
                candidate,
                after,
                verdict,
            )

        monkeypatch.setattr(runner, "project_marginal_speed", project_speed)
        monkeypatch.setattr(
            runner,
            "qualification_authority_digest",
            lambda _value: _d("qualification-authority"),
        )

        self.attempt_reference = EvidenceArtifactRef(
            runner.ATTEMPT_DOMAIN,
            _d("causal-attempt-artifact"),
            2,
            "application/json",
            runner.ATTEMPT_SCHEMA,
        )
        self.published_attempt = None

        def publish_attempt(root, attempt):
            assert root is self.value.evidence_root
            assert type(attempt) is runner.CohortQualificationAttempt
            self.calls.append("attempt.publish")
            self.published_attempt = attempt
            return self.attempt_reference

        def reopen_attempt(root, reference, *, expected):
            assert root is self.value.evidence_root
            assert reference == self.attempt_reference
            assert expected is self.value
            assert self.published_attempt is not None
            self.calls.append("attempt.reopen")
            return self.published_attempt

        monkeypatch.setattr(runner, "publish_causal_qualification", publish_attempt)
        monkeypatch.setattr(runner, "reopen_causal_qualification", reopen_attempt)

    def run(self):
        ids = iter(f"{index + 1:032x}" for index in range(1 + 2 * len(self.value.candidates)))
        reference = runner.run_causal_qualification(
            self.value,
            executor=self.executor,
            entropy_provider=self.entropy_provider,
            hidden_judge=self.hidden_judge,
            deadline=100.0,
            id_factory=lambda: next(ids),
        )
        assert reference == self.attempt_reference
        assert type(self.published_attempt) is runner.CohortQualificationAttempt
        return self.published_attempt


def _discovery_harness(monkeypatch, tmp_path: Path) -> _Harness:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    execution_root = tmp_path / "execution"
    execution_root.mkdir()
    requirement, lifecycle = _discovery_execution(execution_root)
    after_receipts = tuple(
        SimpleNamespace(**row.__dict__, completed_monotonic_s=2.0)
        for row in lifecycle.baseline_after.device_receipts
    )
    lifecycle = replace(
        lifecycle,
        baseline_after=replace(
            lifecycle.baseline_after, device_receipts=after_receipts
        ),
    )
    prepared = lifecycle.prepared
    arm = prepared.source
    reference = _reference()
    context_digest = _d("discovery-calibration-context")
    workload_digest = runner.marginal_workload_digest(
        prepared.baseline_session_plan
    )
    profile = DiscoveryQualificationProfile(
        reference,
        context_digest,
        harness.calibration.digest,
        requirement.digest,
        ("mean_nll", "task_score", "topk_kl"),
        "2",
        prepared.baseline_session_plan.max_new_tokens,
        prepared.baseline_session_plan.top_logprobs_num,
        1,
        _d("support-policy"),
        _d("hidden-task-policy"),
        harness.value.expected_runtime_resource_policy_digest,
        True,
        2,
    )
    authority = runner.DiscoveryCandidateQualificationAuthority(
        arm.selected_delta_digest, profile, requirement
    )
    prompt_digests = runner._planned_prompt_digests(prepared)
    secret_commitment = hashlib.sha256(
        b"optima-selection-secret-v1\0" + harness.secret
    ).hexdigest()
    commitment = SelectionCommitment(
        prepared.source.digest,
        reference.digest,
        workload_digest,
        reference.selection_policy_digest,
        secret_commitment,
        prompt_digests,
        1,
    )
    entropy = SelectionEntropyReceipt(
        commitment.entropy_source_digest,
        commitment.digest,
        _d("discovery-entropy"),
        _d("discovery-entropy-authority"),
    )

    harness.lifecycle = lifecycle
    harness.discovery_requirement = requirement
    harness.discovery_lifecycle = lifecycle
    harness.grades = {}
    harness.commitment = commitment
    harness.entropy = entropy
    harness.hidden_judge.binding = runner.HiddenJudgeBinding(
        reference.hidden_corpus_commitment,
        reference.hidden_judge_digest,
        profile.hidden_task_policy_digest,
    )
    harness.value.prepared = prepared
    harness.value.candidates = (authority,)
    harness.value.commitment = commitment
    harness.value.evidence_root = tmp_path / "quality"

    harness.attempt_reference = EvidenceArtifactRef(
        runner.DISCOVERY_ATTEMPT_DOMAIN,
        _d("discovery-attempt-artifact"),
        2,
        "application/json",
        runner.DISCOVERY_ATTEMPT_SCHEMA,
    )
    harness.published_attempt = None

    def publish_attempt(root, attempt):
        assert root is harness.value.evidence_root
        assert type(attempt) is runner.DiscoveryQualificationAttempt
        harness.published_attempt = attempt
        return harness.attempt_reference

    def reopen_attempt(root, reference, *, expected):
        assert root is harness.value.evidence_root
        assert reference == harness.attempt_reference
        assert expected is harness.value
        return harness.published_attempt

    monkeypatch.setattr(
        runner, "qualification_authority_digest", _REAL_QUALIFICATION_AUTHORITY
    )
    monkeypatch.setattr(runner, "publish_causal_qualification", publish_attempt)
    monkeypatch.setattr(runner, "reopen_causal_qualification", reopen_attempt)
    return harness


def _run_discovery(harness: _Harness):
    ids = iter(f"{index + 1:032x}" for index in range(3))
    reference = runner.run_causal_qualification(
        harness.value,
        executor=harness.executor,
        entropy_provider=harness.entropy_provider,
        hidden_judge=harness.hidden_judge,
        deadline=100.0,
        id_factory=lambda: next(ids),
    )
    assert reference == harness.attempt_reference
    assert type(harness.published_attempt) is runner.DiscoveryQualificationAttempt
    return harness.published_attempt


def test_causal_order_uses_one_multi_candidate_t_lifetime(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS, QualificationDecision.PASS),
        speed=(QualificationDecision.PASS, QualificationDecision.PASS),
        quality=(QualificationDecision.PASS, QualificationDecision.PASS),
    )
    attempt = harness.run()

    assert harness.reference_calls == 1
    assert harness.reference_request_counts == [2]
    assert harness.calls[:8] == [
        "prevalidate",
        "transaction.enter",
        "lifecycle",
        "quiescence.1",
        "entropy",
        "reference",
        "quiescence.2",
        "transaction.exit",
    ]
    assert [row.decision for row in attempt.reports] == [
        QualificationDecision.PASS,
        QualificationDecision.PASS,
    ]


def test_discovery_authority_attempt_roundtrip_and_cross_lane_rejection(
    monkeypatch, tmp_path: Path
) -> None:
    registered = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    ).run()
    harness = _discovery_harness(monkeypatch, tmp_path / "discovery")
    attempt = _run_discovery(harness)

    assert attempt.reports[0].execution_grade.execution_passed
    assert attempt.reports[0].decision is QualificationDecision.PASS
    assert attempt.authority_digest == _REAL_QUALIFICATION_AUTHORITY(harness.value)

    captured: dict[str, object] = {}
    real_digest = runner.canonical_digest

    def capture(domain: str, value: object) -> str:
        if domain == "optima.qualification.discovery-causal-authority":
            captured["payload"] = value
        return real_digest(domain, value)

    monkeypatch.setattr(runner, "canonical_digest", capture)
    assert _REAL_QUALIFICATION_AUTHORITY(harness.value) == attempt.authority_digest
    payload = captured["payload"]

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                key for row in value.values() for key in keys(row)
            }
        if isinstance(value, (list, tuple)):
            return {key for row in value for key in keys(row)}
        return set()

    authority_keys = keys(payload)
    forbidden = (
        "target",
        "catalog",
        "contribution",
        "graph_artifact",
        "graph_evidence",
        "graph_requirement",
    )
    assert not {
        key
        for key in authority_keys
        if any(marker in key for marker in forbidden)
    }

    monkeypatch.setattr(runner, "canonical_digest", real_digest)
    monkeypatch.setattr(runner, "publish_evidence", publish_evidence)
    discovery_root = tmp_path / "durable-discovery"
    discovery_ref = _REAL_PUBLISH_CAUSAL(discovery_root, attempt)
    assert (
        discovery_ref.domain,
        discovery_ref.schema,
        reopen_evidence(discovery_root, discovery_ref),
    ) == (
        runner.DISCOVERY_ATTEMPT_DOMAIN,
        runner.DISCOVERY_ATTEMPT_SCHEMA,
        runner.canonical_json_bytes(attempt.to_dict()),
    )
    assert runner.DiscoveryQualificationAttempt.from_dict(
        runner._canonical_payload(reopen_evidence(discovery_root, discovery_ref))
    ) == attempt

    registered_root = tmp_path / "durable-registered"
    registered_ref = _REAL_PUBLISH_CAUSAL(registered_root, registered)
    with pytest.raises(runner.QualificationRunnerError, match="artifact type"):
        _REAL_REOPEN_CAUSAL(
            registered_root, registered_ref, expected=harness.value
        )


def test_discovery_attempt_rejects_missing_grade_and_wrong_activation_binding(
    monkeypatch, tmp_path: Path
) -> None:
    harness = _discovery_harness(monkeypatch, tmp_path)
    attempt = _run_discovery(harness)
    report = attempt.reports[0]
    grade = report.execution_grade
    requirement = harness.discovery_requirement
    prepared = harness.discovery_lifecycle.prepared.candidates[0]

    missing_grade = attempt.to_dict()
    del missing_grade["reports"][0]["execution_grade"]
    with pytest.raises(runner.QualificationRunnerError, match="fields"):
        runner.DiscoveryQualificationAttempt.from_dict(missing_grade)

    missing_activation = attempt.to_dict()
    missing_activation["reports"][0]["execution_grade"][
        "activation_receipt"
    ] = None
    with pytest.raises(QualificationError, match="lacks activation"):
        runner.DiscoveryQualificationAttempt.from_dict(missing_activation)

    with pytest.raises(QualificationError, match="retained binding"):
        runner.reopen_discovery_execution_binding(
            requirement,
            replace(grade, candidate_lifecycle_digest=_d("wrong lifecycle")),
            prepared,
            candidate_lifecycle_digest=grade.candidate_lifecycle_digest,
            session_id=grade.session_id,
        )

    wrong_receipt = replace(
        grade.activation_receipt,
        overlay_identity_digest=_d("wrong activation overlay"),
    )
    with pytest.raises(QualificationError, match="not authoritative"):
        runner.reopen_discovery_execution_binding(
            requirement,
            replace(grade, activation_receipt=wrong_receipt),
            prepared,
            candidate_lifecycle_digest=grade.candidate_lifecycle_digest,
            session_id=grade.session_id,
        )


def test_candidate_headlines_recompute_pass_fail_and_no_decision(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,) * 3,
        speed=(
            QualificationDecision.PASS,
            QualificationDecision.FAIL,
            QualificationDecision.PASS,
        ),
        quality=(
            QualificationDecision.PASS,
            QualificationDecision.PASS,
            QualificationDecision.NO_DECISION,
        ),
    )

    reports = harness.run().reports
    assert tuple(row.decision for row in reports) == (
        QualificationDecision.PASS,
        QualificationDecision.FAIL,
        QualificationDecision.NO_DECISION,
    )
    assert tuple(row.retryable for row in reports) == (False, False, True)
    with pytest.raises(runner.QualificationRunnerError, match="headline"):
        runner.CandidateQualificationReport(
            **{**reports[0].__dict__, "decision": QualificationDecision.FAIL}
        )


def test_t_exchange_substitution_is_rejected(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS, QualificationDecision.PASS),
        speed=(QualificationDecision.PASS, QualificationDecision.PASS),
        quality=(QualificationDecision.PASS, QualificationDecision.PASS),
        swap_exchanges=True,
    )
    with pytest.raises(runner.QualificationRunnerError, match="request|exchange"):
        harness.run()


def test_pre_t_quiescence_failure_prevents_reference_launch(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
        fail_pre_t_quiescence=True,
    )
    with pytest.raises(runner.QualificationRunnerError, match="quiescence"):
        harness.run()
    assert harness.reference_calls == 0
    assert "entropy" not in harness.calls


def test_stale_hidden_judge_binding_is_rejected_before_b(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    )
    harness.hidden_judge.binding = runner.HiddenJudgeBinding(
        _d("stale-corpus"),
        _d("hidden-judge"),
        _d("hidden-task-policy"),
    )

    with pytest.raises(runner.QualificationRunnerError, match="hidden judge authority"):
        harness.run()
    assert "lifecycle" not in harness.calls
    assert harness.reference_calls == 0


def test_identical_hidden_judge_inputs_are_memoized_across_the_cohort(monkeypatch) -> None:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS, QualificationDecision.PASS),
        speed=(QualificationDecision.PASS, QualificationDecision.PASS),
        quality=(QualificationDecision.PASS, QualificationDecision.PASS),
        exercise_judge_cache=True,
    )
    harness.run()
    assert harness.hidden_judge.calls == 1


@pytest.mark.parametrize("mislabeled", ("prompt", "output"))
def test_hidden_judge_receipt_cannot_relabel_prompt_or_output(mislabeled) -> None:
    prompt_digest = _d("judge-prompt")
    output_ids = (1,)
    binding = runner.HiddenJudgeBinding(
        _d("hidden-corpus"),
        _d("hidden-judge"),
        _d("hidden-task-policy"),
    )
    profile = SimpleNamespace(
        reference=SimpleNamespace(
            hidden_corpus_commitment=binding.hidden_corpus_commitment,
            hidden_judge_digest=binding.hidden_judge_digest,
        ),
        hidden_task_policy_digest=binding.hidden_task_policy_digest,
        hidden_tasks_per_prompt=1,
    )

    class MislabeledJudge:
        def __init__(self) -> None:
            self.binding = binding

        def __call__(self, *, prompt_digest, output_ids, task_digests):
            receipt_prompt = _d("other-prompt") if mislabeled == "prompt" else prompt_digest
            output_digest = runner.hidden_judge_output_digest(prompt_digest, output_ids)
            if mislabeled == "output":
                output_digest = _d("other-output")
            return runner.HiddenJudgeReceipt(
                self.binding.digest,
                receipt_prompt,
                output_digest,
                task_digests,
                (True,) * len(task_digests),
            )

    with pytest.raises(runner.QualificationRunnerError):
        runner._rollout(
            profile=profile,
            prompt_digest=prompt_digest,
            frame={"top_logprobs": (((-0.1, 1), (-1.0, 2)),)},
            role_input=ReferenceRoleInput(output_ids, ((1, 2),)),
            role_evidence=SimpleNamespace(
                tokens=(ReferenceTokenEvidence(-0.25, 1, (-0.1, -1.0)),)
            ),
            hidden_judge=MislabeledJudge(),
        )


def test_shared_manager_reservation_excludes_another_thread() -> None:
    lock = threading.RLock()
    first = object.__new__(OCIEngineExecutor)
    second = object.__new__(OCIEngineExecutor)
    first._lock = second._lock = lock
    entered, release = threading.Event(), threading.Event()

    def hold() -> None:
        with first.exclusive_transaction():
            entered.set()
            assert release.wait(timeout=2)

    thread = threading.Thread(target=hold)
    thread.start()
    assert entered.wait(timeout=2)
    try:
        with pytest.raises(OCIBackendError, match="active transaction"):
            with second.exclusive_transaction():
                raise AssertionError("another thread entered the reserved manager")
    finally:
        release.set()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_reports_and_attempt_expose_no_score_crown_or_settlement_fields(monkeypatch) -> None:
    attempt = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    ).run()

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {key for row in value.values() for key in keys(row)}
        if isinstance(value, list):
            return {key for row in value for key in keys(row)}
        return set()

    emitted = keys(attempt.to_dict())
    assert not ({"score", "crown", "crownable", "settlement", "winner"} & emitted)


def test_registered_authority_digest_and_report_wire_format_remain_stable(
    monkeypatch, tmp_path: Path
) -> None:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    )
    attempt = harness.run()
    value = harness.value
    captured: dict[str, object] = {}
    real_digest = runner.canonical_digest

    def capture(domain: str, payload: object) -> str:
        if domain == "optima.qualification.causal-authority":
            captured["payload"] = payload
        return real_digest(domain, payload)

    monkeypatch.setattr(runner, "CandidateQualificationAuthority", SimpleNamespace)
    monkeypatch.setattr(runner, "canonical_digest", capture)
    authority_digest = _REAL_QUALIFICATION_AUTHORITY(value)
    assert authority_digest == real_digest(
        "optima.qualification.causal-authority", captured["payload"]
    )
    authority_payload = captured["payload"]
    assert set(authority_payload) == {
        "calibration",
        "candidates",
        "commitment",
        "incumbent_preflight",
        "model_mount",
        "policies",
        "reference",
        "source",
    }
    assert set(authority_payload["candidates"][0]) == {
        "arm",
        "delta",
        "graph_artifact",
        "graph_evidence_ref",
        "graph_requirement",
        "launch",
        "native",
        "preflight",
        "profile",
        "target",
    }
    monkeypatch.setattr(runner, "canonical_digest", real_digest)

    report_fields = (
        "selected_delta_digest",
        "marginal_arm_digest",
        "candidate_launch_digest",
        "target_id",
        "profile_digest",
        "calibration_digest",
        "graph_grade_digest",
        "graph_decision",
        "speed_evidence_digest",
        "speed_decision",
        "speedup",
        "quality_evidence_digest",
        "quality_decision",
        "candidate_mean_teacher_nll",
        "raw_quality_artifact",
        "raw_quality_binding",
        "speed_witness",
        "t_request_sha256",
        "decision",
        "reason",
        "retryable",
    )
    report = attempt.reports[0]
    assert tuple(report.to_dict()) == report_fields
    assert runner.canonical_json_bytes(report.to_dict()) == runner.canonical_json_bytes(
        {
            field: runner._encode_record(getattr(report, field))
            for field in report_fields
        }
    )
    assert tuple(attempt.to_dict()) == (
        "authority_digest",
        "source_digest",
        "cohort_trajectory_digest",
        "commitment",
        "teardown_before_t",
        "entropy",
        "entropy_observed_monotonic_s",
        "selection",
        "reference_execution",
        "teardown_after_t",
        "reports",
    )

    monkeypatch.setattr(runner, "publish_evidence", publish_evidence)
    root = tmp_path / "registered-wire"
    reference = _REAL_PUBLISH_CAUSAL(root, attempt)
    assert (reference.domain, reference.schema) == (
        runner.ATTEMPT_DOMAIN,
        runner.ATTEMPT_SCHEMA,
    )
    assert reopen_evidence(root, reference) == runner.canonical_json_bytes(
        attempt.to_dict()
    )


def test_durable_attempt_roundtrip_rejects_nested_decision_tamper(
    monkeypatch, tmp_path
) -> None:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    )
    attempt = harness.run()
    root = tmp_path / "attempt-evidence"

    # Restore the real byte publisher only after the fake causal run is complete.
    monkeypatch.setattr(runner, "publish_evidence", publish_evidence)
    reference = _REAL_PUBLISH_CAUSAL(root, attempt)
    reopened = runner.CohortQualificationAttempt.from_dict(
        runner._canonical_payload(reopen_evidence(root, reference))
    )
    assert reopened == attempt

    tampered = attempt.to_dict()
    tampered["reports"][0]["decision"] = QualificationDecision.FAIL.value
    forged = publish_evidence(
        root,
        runner.canonical_json_bytes(tampered),
        domain=runner.ATTEMPT_DOMAIN,
        media_type="application/json",
        schema=runner.ATTEMPT_SCHEMA,
    )
    with pytest.raises(runner.QualificationRunnerError, match="headline"):
        _REAL_REOPEN_CAUSAL(root, forged, expected=harness.value)


def test_reopen_rejects_self_consistent_speed_witness_arm_relabel(
    monkeypatch, tmp_path
) -> None:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    )
    attempt = harness.run()
    payload = attempt.to_dict()
    report = payload["reports"][0]
    speed = report["speed_witness"]
    speed["selected_delta_digest"] = _d("relabeled-delta")
    rates = tuple(runner._rate_from_dict(row) for row in speed["rates"])
    speed["evidence_digest"] = runner._projection_digest(
        speed["selected_delta_digest"],
        speed["candidate_launch_digest"],
        speed["calibration_digest"],
        speed["calibration_context_digest"],
        speed["workload_digest"],
        speed["runtime_resource_policy_digest"],
        rates,
    )
    report["speed_evidence_digest"] = speed["evidence_digest"]

    root = tmp_path / "speed-relabel"
    forged = publish_evidence(
        root,
        runner.canonical_json_bytes(payload),
        domain=runner.ATTEMPT_DOMAIN,
        media_type="application/json",
        schema=runner.ATTEMPT_SCHEMA,
    )
    monkeypatch.setattr(runner, "_validate_reference_execution", lambda *_args: None)
    monkeypatch.setattr(
        runner,
        "reopen_calibration_evidence",
        lambda *_args, **_kwargs: harness.calibration,
    )
    monkeypatch.setattr(
        runner,
        "reopen_graph_verification",
        lambda *_args, **_kwargs: next(iter(harness.grades.values())),
    )
    monkeypatch.setattr(
        runner, "reopen_reference_quality_evidence", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        runner,
        "score_reference_quality",
        lambda *_args, **_kwargs: _quality_verdict(
            QualificationDecision.PASS, 0, harness.calibration.digest
        ),
    )
    with pytest.raises(runner.QualificationRunnerError, match="speed witness"):
        _REAL_REOPEN_CAUSAL(root, forged, expected=harness.value)


@pytest.mark.parametrize("tamper", ("launch", "causal_time"))
def test_reference_execution_witness_rejects_launch_and_causal_tamper(
    monkeypatch, tamper
) -> None:
    harness = _Harness(
        monkeypatch,
        graph=(QualificationDecision.PASS,),
        speed=(QualificationDecision.PASS,),
        quality=(QualificationDecision.PASS,),
    )
    attempt = harness.run()
    witness = attempt.reference_execution
    identity = SimpleNamespace(
        runtime_digest=_d("reference-runtime"),
        base_engine_digest=_d("reference-base"),
        validator_overlay_digest=_d("reference-overlay"),
    )
    monkeypatch.setattr(runner, "runtime_identity_from_preflight", lambda _row: identity)
    harness.value.pristine_binding = SimpleNamespace(
        runtime_preflight_receipt=SimpleNamespace(
            sha256=witness.runtime_preflight_receipt_sha256
        ),
        native_build_spec=SimpleNamespace(digest=witness.build_spec_digest),
    )
    harness.value.pristine_launch = SimpleNamespace(
        digest=witness.launch_digest,
        hardware=SimpleNamespace(visible_gpu_count=1),
    )
    harness.value.model_mount = SimpleNamespace(digest=witness.arena_model_receipt_digest)
    harness.value.pristine_stack = SimpleNamespace(digest=_d("pristine-stack"))
    harness.value.reference_engine_config = SimpleNamespace(digest=_d("reference-config"))
    harness.value.reference_preflight = SimpleNamespace(digest=_d("reference-preflight"))
    runtime_digest = runner.canonical_digest(
        "optima.qualification.reference-runtime",
        {
            "base": identity.base_engine_digest,
            "runtime": identity.runtime_digest,
            "validator_overlay": identity.validator_overlay_digest,
        },
    )
    plan_digest = runner.canonical_digest(
        "optima.eval.reference-session-plan",
        {
            "engine_config_digest": harness.value.reference_engine_config.digest,
            "expected_preflight_digest": harness.value.reference_preflight.digest,
            "pristine_stack_digest": harness.value.pristine_stack.digest,
            "reference_manifest_digest": harness.value.candidates[0].profile.reference.digest,
            "request_plan_digest": witness.request_plan_digest,
            "request_sha256": list(witness.request_sha256),
        },
    )
    session_digest = runner.canonical_digest(
        "optima.eval.pristine-reference-session",
        {
            "exchanges": [
                {
                    "request_index": index,
                    "request_sha256": request,
                    "evidence_frame_sha256": evidence,
                }
                for index, (request, evidence) in enumerate(
                    zip(witness.request_sha256, witness.evidence_frame_sha256, strict=True)
                )
            ],
            "launch_digest": witness.launch_digest,
            "preflight_digest": harness.value.reference_preflight.digest,
            "reference_manifest_digest": harness.value.candidates[0].profile.reference.digest,
            "request_plan_digest": witness.request_plan_digest,
            "schema": "optima.pristine-reference-session.v1",
            "session_id": witness.session_id,
            "session_plan_digest": plan_digest,
        },
    )
    honest_witness = replace(
        witness,
        runtime_identity_digest=runtime_digest,
        plan_digest=plan_digest,
        session_digest=session_digest,
    )
    honest = replace(attempt, reference_execution=honest_witness)
    runner._validate_reference_execution(honest, harness.value)

    if tamper == "launch":
        forged_witness = replace(honest_witness, launch_digest=_d("other-launch"))
    else:
        post = list(honest_witness.device_receipts[1])
        post[8] = "7"
        forged_witness = replace(
            honest_witness,
            device_receipts=(honest_witness.device_receipts[0], tuple(post)),
        )
    with pytest.raises(runner.QualificationRunnerError, match="pristine T"):
        runner._validate_reference_execution(
            replace(honest, reference_execution=forged_witness), harness.value
        )
