from __future__ import annotations

import json
import os
from dataclasses import replace
from decimal import Decimal, localcontext

import pytest

import optima.eval.reference_quality as reference_quality
from optima.eval.calibration import (
    CalibrationContext,
    CalibrationControl,
    CalibrationManifest,
    MetricCalibration,
    SpeedCalibration,
)
from optima.eval.evidence_store import (
    publish_canonical_json_evidence,
    publish_evidence,
)
from optima.eval.reference_quality import (
    HiddenTaskEvidence,
    NormalizedTopKDistribution,
    PromptQualityEvidence,
    RAW_QUALITY_DOMAIN,
    RAW_QUALITY_SCHEMA,
    RawHiddenTaskResult,
    RawPromptQualityEvidence,
    RawRolloutEvidence,
    RawTokenEvidence,
    ReferenceQualityError,
    ReferenceQualityEvidence,
    ReferenceQualityRawArtifact,
    ReferenceQualityRawBinding,
    ReferenceQualityVerdict,
    RolloutKLEvidence,
    RolloutQualityEvidence,
    TeacherNLLEvidence,
    TokenProbability,
    hidden_task_plan_digest,
    distribution_from_f32_logprobs,
    raw_trajectory_projection_digest,
    reopen_reference_quality_evidence,
    score_reference_quality,
    target_nll_from_f32,
)
from optima.stack_identity import canonical_json_bytes


def _digest(character: str) -> str:
    return character * 64


def _context() -> CalibrationContext:
    return CalibrationContext(*[character * 64 for character in "123456789ab"])


def _calibration(*, status: str = "frozen", z: str = "0.000001") -> CalibrationManifest:
    return CalibrationManifest(
        context=_context(),
        algorithm_id="teacher-familywise-v1",
        status=status,
        speed=SpeedCalibration("0.005", "2", "0.1"),
        quality_metrics=(
            MetricCalibration("mean_nll", "lower", "0.1", "0.1"),
            MetricCalibration("task_score", "higher", "0.05", "0.05", "0.8"),
            MetricCalibration("topk_kl", "lower", "0.02", "0.03"),
        ),
        familywise_z=z,
        raw_evidence_digest=_digest("c"),
        seed_digests=(_digest("d"),),
        controls=(
            CalibrationControl("negative", _digest("e"), _digest("f"), "FAIL"),
            CalibrationControl("positive", _digest("1"), _digest("2"), "PASS"),
            CalibrationControl("stock", _digest("3"), _digest("4"), "PASS"),
        ),
    )


def _rollout(
    *, nll: str = "1", kl: str = "0.01", task: str = "9", tokens: int = 10
) -> RolloutQualityEvidence:
    return RolloutQualityEvidence(
        teacher_nll=TeacherNLLEvidence(tokens, str(float(nll) * tokens).rstrip("0").rstrip("."), "2", 0),
        rollout_kl=RolloutKLEvidence(tokens, kl, "0.02", "0.02", 0, "0.01"),
        hidden_task=HiddenTaskEvidence(task, 10),
    )


def _prompt(
    identity: str,
    *,
    baseline: RolloutQualityEvidence | None = None,
    candidate: RolloutQualityEvidence | None = None,
    control: RolloutQualityEvidence | None = None,
    matches: int = 10,
) -> PromptQualityEvidence:
    return PromptQualityEvidence(
        _digest(identity),
        baseline or _rollout(),
        candidate or _rollout(),
        control or _rollout(),
        matches,
        10,
    )


def _evidence(
    calibration: CalibrationManifest,
    prompts: tuple[PromptQualityEvidence, ...] | None = None,
) -> ReferenceQualityEvidence:
    return ReferenceQualityEvidence(
        reference_manifest_digest=_context().reference_manifest_digest,
        calibration_digest=calibration.digest,
        raw_evidence_digest=_digest("5"),
        prompts=prompts or (_prompt("6"), _prompt("7")),
        hidden_tasks_present=True,
    )


def test_quality_round_trip_preserves_distinct_nll_and_rollout_kl() -> None:
    calibration = _calibration()
    evidence = _evidence(calibration)
    reopened = ReferenceQualityEvidence.from_dict(evidence.to_dict())
    assert reopened == evidence
    assert reopened.digest == evidence.digest
    prompt = reopened.prompts[0].candidate
    assert type(prompt.teacher_nll) is TeacherNLLEvidence
    assert type(prompt.rollout_kl) is RolloutKLEvidence
    assert prompt.teacher_nll.mean == 1.0
    assert prompt.rollout_kl.mean_kl == "0.01"


def test_faithful_candidate_passes_frozen_familywise_policy() -> None:
    calibration = _calibration()
    verdict = score_reference_quality(
        _evidence(calibration), calibration=calibration, expected_context=_context()
    )
    assert verdict.decision == "PASS"
    assert verdict.candidate_mean_teacher_nll == "1"
    assert not verdict.failed_metrics and not verdict.overlapping_metrics


def test_teacher_nll_regression_fails() -> None:
    calibration = _calibration()
    bad = _rollout(nll="1.5")
    prompts = (_prompt("6", candidate=bad), _prompt("7", candidate=bad))
    verdict = score_reference_quality(
        _evidence(calibration, prompts), calibration=calibration, expected_context=_context()
    )
    assert verdict.decision == "FAIL"
    assert "mean_nll" in verdict.failed_metrics


def test_stock_control_drift_is_no_decision_not_candidate_failure() -> None:
    calibration = _calibration()
    drift = _rollout(nll="1.5")
    prompts = (_prompt("6", control=drift), _prompt("7", control=drift))
    verdict = score_reference_quality(
        _evidence(calibration, prompts), calibration=calibration, expected_context=_context()
    )
    assert verdict.decision == "NO_DECISION"
    assert "mean_nll.stock_drift" in verdict.overlapping_metrics
    assert not verdict.failed_metrics


def test_hidden_task_floor_is_an_external_failure() -> None:
    calibration = _calibration()
    low = _rollout(task="5")
    prompts = (_prompt("6", candidate=low), _prompt("7", candidate=low))
    verdict = score_reference_quality(
        _evidence(calibration, prompts), calibration=calibration, expected_context=_context()
    )
    assert verdict.decision == "FAIL"
    assert "task_score.absolute_floor" in verdict.failed_metrics


def test_provisional_calibration_can_never_pass() -> None:
    calibration = _calibration(status="provisional")
    verdict = score_reference_quality(
        _evidence(calibration), calibration=calibration, expected_context=_context()
    )
    assert verdict.decision == "NO_DECISION"
    assert verdict.overlapping_metrics == ("calibration.provisional",)


def test_exact_token_diagnostic_cannot_change_the_grade() -> None:
    calibration = _calibration()
    exact = _evidence(calibration)
    mismatched = replace(
        exact,
        prompts=tuple(replace(prompt, exact_token_matches=0) for prompt in exact.prompts),
    )
    assert exact.digest != mismatched.digest
    left = score_reference_quality(exact, calibration=calibration, expected_context=_context())
    right = score_reference_quality(
        mismatched, calibration=calibration, expected_context=_context()
    )
    assert left.decision == right.decision == "PASS"


def test_swapped_calibration_or_reference_rejects() -> None:
    calibration = _calibration()
    evidence = _evidence(calibration)
    with pytest.raises(ReferenceQualityError, match="another calibration"):
        score_reference_quality(
            replace(evidence, calibration_digest=_digest("8")),
            calibration=calibration,
            expected_context=_context(),
        )
    with pytest.raises(ReferenceQualityError, match="another pristine reference"):
        score_reference_quality(
            replace(evidence, reference_manifest_digest=_digest("8")),
            calibration=calibration,
            expected_context=_context(),
        )


def test_prompt_identity_must_be_unique_and_sorted() -> None:
    calibration = _calibration()
    evidence = _evidence(calibration)
    with pytest.raises(ReferenceQualityError, match="prompts"):
        replace(evidence, prompts=tuple(reversed(evidence.prompts)))
    with pytest.raises(ReferenceQualityError, match="prompts"):
        replace(evidence, prompts=(evidence.prompts[0], evidence.prompts[0]))


def test_hidden_task_coverage_is_all_or_none() -> None:
    calibration = _calibration()
    no_hidden = replace(_rollout(), hidden_task=HiddenTaskEvidence("0", 0))
    prompt = _prompt("6", baseline=no_hidden, candidate=no_hidden, control=no_hidden)
    evidence = ReferenceQualityEvidence(
        _context().reference_manifest_digest,
        calibration.digest,
        _digest("5"),
        (prompt,),
        False,
    )
    assert not evidence.hidden_tasks_present
    with pytest.raises(ReferenceQualityError, match="declaration/coverage"):
        replace(evidence, hidden_tasks_present=True)


def test_strict_parser_rejects_extra_fields_and_noncanonical_decimals() -> None:
    calibration = _calibration()
    raw = _evidence(calibration).to_dict()
    raw["decision"] = "PASS"
    with pytest.raises(ReferenceQualityError, match="fields do not match"):
        ReferenceQualityEvidence.from_dict(raw)
    with pytest.raises(ReferenceQualityError, match="canonical"):
        TeacherNLLEvidence(10, "10.0", "2", 0)


def _raw_binding(
    calibration: CalibrationManifest,
    prompts: tuple[RawPromptQualityEvidence, ...],
) -> ReferenceQualityRawBinding:
    return ReferenceQualityRawBinding(
        qualification_identity_digest=_digest("6"),
        reference_manifest_digest=_context().reference_manifest_digest,
        calibration_digest=calibration.digest,
        selection_digest=_digest("7"),
        candidate_lifecycle_digest=_digest("8"),
        selected_trajectory_digest=_digest("a"),
        selected_trajectory_projection_digest=raw_trajectory_projection_digest(prompts),
        selected_prompt_digests=tuple(row.prompt_digest for row in prompts),
        t_session_digest=_digest("9"),
        t_request_sha256=_digest("c"),
        support_policy_digest=_digest("b"),
        hidden_task_plan_digest=hidden_task_plan_digest(prompts),
        nll_tail_threshold="2",
        tokens_per_prompt=2,
        topk_width=2,
        hidden_tasks_per_prompt=2,
    )


def _dist(
    left: str, right: str, tail: str = "0.1", argmax: int | None = None
) -> NormalizedTopKDistribution:
    argmax = (1 if Decimal(left) > Decimal(right) else 2) if argmax is None else argmax
    return NormalizedTopKDistribution(
        (TokenProbability(1, left), TokenProbability(2, right)), tail, argmax
    )


def _raw_rollout(
    *,
    nlls=("1", "2"),
    token_ids=(1, 2),
    rollout=(("0.7", "0.2"), ("0.7", "0.2")),
    task_passes=(True, False),
) -> RawRolloutEvidence:
    teacher = _dist("0.7", "0.2")
    tokens = tuple(
        RawTokenEvidence(
            index,
            token_ids[index],
            nlls[index],
            teacher,
            _dist(*rollout[index]),
        )
        for index in range(2)
    )
    tasks = tuple(
        RawHiddenTaskResult(_digest(str(index + 1)), passed)
        for index, passed in enumerate(task_passes)
    )
    return RawRolloutEvidence(tokens, tasks)


def _raw_artifact(calibration: CalibrationManifest) -> ReferenceQualityRawArtifact:
    baseline = _raw_rollout()
    candidate = _raw_rollout(
        nlls=("1.5", "3"),
        token_ids=(1, 3),
        rollout=(("0.6", "0.3"), ("0.2", "0.7")),
        task_passes=(True, True),
    )
    prompts = (RawPromptQualityEvidence(_digest("6"), baseline, candidate, baseline),)
    return ReferenceQualityRawArtifact(_raw_binding(calibration, prompts), prompts)


def _publish(root, artifact: ReferenceQualityRawArtifact):
    return publish_canonical_json_evidence(
        root,
        artifact.to_dict(),
        domain=RAW_QUALITY_DOMAIN,
        schema=RAW_QUALITY_SCHEMA,
    )


def test_authoritative_raw_artifact_recomputes_every_summary_and_is_deterministic(tmp_path):
    calibration = _calibration()
    artifact = _raw_artifact(calibration)
    reference = _publish(tmp_path / "evidence", artifact)
    first = reopen_reference_quality_evidence(
        tmp_path / "evidence", reference, expected_binding=artifact.binding
    )
    second = reopen_reference_quality_evidence(
        tmp_path / "evidence", reference, expected_binding=artifact.binding
    )
    with localcontext() as context:
        context.prec = 6
        altered_context = reopen_reference_quality_evidence(
            tmp_path / "evidence", reference, expected_binding=artifact.binding
        )

    assert first == second == altered_context
    assert first.raw_evidence_digest == reference.sha256
    prompt = first.prompts[0]
    assert prompt.candidate.teacher_nll == TeacherNLLEvidence(2, "4.5", "3", 1)
    assert float(prompt.candidate.rollout_kl.mean_kl) > 0
    assert prompt.candidate.rollout_kl.argmax_disagreements == 1
    assert prompt.candidate.rollout_kl.mean_coverage_deviation == "0"
    assert prompt.candidate.hidden_task == HiddenTaskEvidence("2", 2)
    assert (prompt.exact_token_matches, prompt.exact_token_total) == (1, 2)
    assert ReferenceQualityRawArtifact.from_dict(artifact.to_dict()) == artifact


def test_raw_artifact_rejects_summaries_unknown_fields_floats_and_noncanonical_bytes(tmp_path):
    calibration = _calibration()
    artifact = _raw_artifact(calibration)
    asserted = artifact.to_dict()
    asserted["teacher_nll"] = {"nll_sum": "0"}
    with pytest.raises(ReferenceQualityError, match="fields"):
        ReferenceQualityRawArtifact.from_dict(asserted)

    floated = artifact.to_dict()
    floated["prompts"][0]["candidate"]["tokens"][0]["target_nll"] = 1.0
    reference = publish_evidence(
        tmp_path / "floats",
        json.dumps(floated, sort_keys=True, separators=(",", ":")).encode(),
        domain=RAW_QUALITY_DOMAIN,
        media_type="application/json",
        schema=RAW_QUALITY_SCHEMA,
    )
    with pytest.raises(ReferenceQualityError, match="float"):
        reopen_reference_quality_evidence(
            tmp_path / "floats", reference, expected_binding=artifact.binding
        )

    payload = canonical_json_bytes(artifact.to_dict()) + b"\n"
    reference = publish_evidence(
        tmp_path / "noncanonical",
        payload,
        domain=RAW_QUALITY_DOMAIN,
        media_type="application/json",
        schema=RAW_QUALITY_SCHEMA,
    )
    with pytest.raises(ReferenceQualityError, match="canonically"):
        reopen_reference_quality_evidence(
            tmp_path / "noncanonical", reference, expected_binding=artifact.binding
        )


def test_reopen_rejects_relabel_tamper_truncation_and_wrong_artifact_type(tmp_path):
    calibration = _calibration()
    artifact = _raw_artifact(calibration)
    root = tmp_path / "evidence"
    reference = _publish(root, artifact)

    relabeled = replace(artifact.binding, selection_digest=_digest("a"))
    with pytest.raises(ReferenceQualityError, match="relabeled"):
        reopen_reference_quality_evidence(root, reference, expected_binding=relabeled)
    with pytest.raises(ReferenceQualityError, match="type"):
        reopen_reference_quality_evidence(
            root, replace(reference, schema="untrusted.v1"), expected_binding=artifact.binding
        )

    target = root / reference.domain / reference.sha256[:2] / reference.sha256
    os.chmod(target, 0o600)
    target.write_bytes(target.read_bytes()[:-1])
    os.chmod(target, 0o400)
    with pytest.raises(ReferenceQualityError, match="cannot reopen"):
        reopen_reference_quality_evidence(root, reference, expected_binding=artifact.binding)


def test_raw_token_contract_rejects_bad_support_normalization_order_and_bounds():
    teacher = _dist("0.7", "0.2")
    with pytest.raises(ReferenceQualityError, match="normalize"):
        NormalizedTopKDistribution(
            (TokenProbability(1, "0.6"), TokenProbability(2, "0.2")), "0.1", 1
        )
    with pytest.raises(ReferenceQualityError, match="support"):
        RawTokenEvidence(
            0,
            1,
            "1",
            teacher,
            NormalizedTopKDistribution(
                (TokenProbability(1, "0.7"), TokenProbability(3, "0.2")), "0.1", 1
            ),
        )
    with pytest.raises(ReferenceQualityError, match="positions"):
        replace(_raw_rollout(), tokens=tuple(reversed(_raw_rollout().tokens)))
    with pytest.raises(ReferenceQualityError, match="bounded"):
        TokenProbability(1, "0." + "1" * 96)


def test_retained_support_projection_is_exact_deterministic_and_allows_ties():
    first = distribution_from_f32_logprobs((3, 7), (-2.0, -2.0), true_argmax_token_id=11)
    second = distribution_from_f32_logprobs((3, 7), (-2.0, -2.0), true_argmax_token_id=11)
    assert first == second
    assert first.entries[0].probability == first.entries[1].probability
    assert first.true_argmax_token_id == 11
    assert sum(Decimal(row.probability) for row in first.entries) + Decimal(
        first.tail_probability
    ) == 1
    saturated = distribution_from_f32_logprobs((3, 7), (0.0, 0.0), true_argmax_token_id=3)
    assert Decimal(saturated.tail_probability) == Decimal(1) / Decimal(10**18)
    assert target_nll_from_f32(-2.5) == "2.5"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("tokens_per_prompt", 1),
        ("topk_width", 1),
        ("hidden_tasks_per_prompt", 1),
        ("hidden_task_plan_digest", _digest("e")),
    ),
)
def test_raw_artifact_cannot_shrink_registered_quality_coverage(field, value):
    calibration = _calibration()
    artifact = _raw_artifact(calibration)
    with pytest.raises(ReferenceQualityError, match="coverage|task plan"):
        replace(artifact, binding=replace(artifact.binding, **{field: value}))


def test_raw_artifact_cannot_substitute_selected_prompts_or_retained_trajectory():
    artifact = _raw_artifact(_calibration())
    prompt = artifact.prompts[0]
    with pytest.raises(ReferenceQualityError, match="bound selection"):
        replace(artifact, prompts=(replace(prompt, prompt_digest=_digest("d")),))

    tokens = prompt.candidate.tokens
    substituted = replace(
        prompt.candidate,
        tokens=(replace(tokens[0], token_id=999), *tokens[1:]),
    )
    with pytest.raises(ReferenceQualityError, match="retained B/C/B-prime"):
        replace(artifact, prompts=(replace(prompt, candidate=substituted),))

    alternate = NormalizedTopKDistribution(
        (TokenProbability(1, "0.7"), TokenProbability(3, "0.2")), "0.1", 1
    )
    substituted = replace(
        prompt.candidate,
        tokens=(replace(tokens[0], teacher_topk=alternate, rollout_topk=alternate), *tokens[1:]),
    )
    with pytest.raises(ReferenceQualityError, match="retained B/C/B-prime"):
        replace(artifact, prompts=(replace(prompt, candidate=substituted),))

    substituted = replace(
        prompt.candidate,
        tokens=(replace(tokens[0], rollout_topk=replace(
            tokens[0].rollout_topk, true_argmax_token_id=2
        )), *tokens[1:]),
    )
    with pytest.raises(ReferenceQualityError, match="retained B/C/B-prime"):
        replace(artifact, prompts=(replace(prompt, candidate=substituted),))


def test_hidden_task_relabel_and_inconsistent_verdict_are_rejected():
    baseline = _raw_rollout()
    relabeled = replace(
        baseline,
        hidden_tasks=(RawHiddenTaskResult(_digest("a"), True),),
    )
    with pytest.raises(ReferenceQualityError, match="identities"):
        RawPromptQualityEvidence(_digest("6"), baseline, baseline, relabeled)

    with pytest.raises(ReferenceQualityError, match="disagrees"):
        ReferenceQualityVerdict(
            "PASS", ("mean_nll",), (), "1", _digest("a"), _digest("b")
        )
    with pytest.raises(ReferenceQualityError, match="finite bound"):
        TeacherNLLEvidence(1, "9" * 19, "1", 0)


def test_nonfinite_computed_familywise_bounds_reject(monkeypatch):
    calibration = _calibration()
    monkeypatch.setattr(reference_quality.statistics, "stdev", lambda _values: float("inf"))
    with pytest.raises(ReferenceQualityError, match="nonfinite"):
        score_reference_quality(
            _evidence(calibration), calibration=calibration, expected_context=_context()
        )
