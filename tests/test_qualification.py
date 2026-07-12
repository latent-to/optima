from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from optima.eval.qualification import (
    GRAPH_EVIDENCE_DOMAIN,
    GRAPH_EVIDENCE_MEDIA_TYPE,
    GRAPH_EVIDENCE_SCHEMA,
    GraphMemberEvidence,
    GraphShapeEvidence,
    GraphVariantEvidence,
    GraphVariantRequirement,
    GraphVerificationBinding,
    GraphVerificationEvidenceRef,
    GraphVerificationGrade,
    GraphVerificationMemberBinding,
    GraphVerificationRawEvidence,
    GraphVerificationRequirement,
    QualificationProfile,
    QualificationDecision,
    QualificationError,
    ReferenceManifest,
    SelectionCommitment,
    SelectionEntropyReceipt,
    SelectionReceipt,
    cohort_trajectory_digest,
    derived_hidden_task_plan_digest,
    lifecycle_prompt_digests,
    reopen_graph_verification,
    regrade_graph_verification,
    selected_trajectory_digest,
    selected_trajectory_projection_digest,
    validate_quality_binding,
)
from optima.eval.evidence_store import publish_evidence
from optima.stack_identity import canonical_json_bytes


def _d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _reference() -> ReferenceManifest:
    return ReferenceManifest(*(_d(f"reference:{index}") for index in range(18)))


def _member(slot: str) -> GraphVerificationMemberBinding:
    return GraphVerificationMemberBinding(
        slot,
        _d(slot + ":target"),
        _d(slot + ":contract"),
        slot + ".verify.v1",
    )


def _requirement(*, atomic: bool = True) -> GraphVerificationRequirement:
    slots = (
        (
            "collective.ar_residual_rmsnorm",
            "collective.moe_finalize_ar_rmsnorm",
        )
        if atomic
        else ("activation.silu_and_mul",)
    )
    members = tuple(_member(slot) for slot in slots)
    binding = GraphVerificationBinding(
        marginal_arm_digest=_d("arm"),
        candidate_launch_digest=_d("launch"),
        contribution_ref_digest=_d("contribution"),
        selected_delta_digest=_d("delta"),
        target_id="collective.moe_epilogue.v1" if atomic else slots[0],
        target_spec_digest=_d("selected-target"),
        catalog_digest=_d("catalog"),
        members=members,
        verification_policy_digest=_d("verification-policy"),
    )
    variants = []
    for slot in slots:
        shapes = tuple(sorted((_d(slot + ":shape-a"), _d(slot + ":shape-b"))))
        variants.append(GraphVariantRequirement(slot, "default", shapes, True, shapes))
    return GraphVerificationRequirement(binding, tuple(variants), 3)


def _shape(
    descriptor: str,
    failure: str = "none",
    *,
    replays: int | None = None,
) -> GraphShapeEvidence:
    states = {
        "none": (True, True, True, True, 3),
        "not_applicable": (False, False, False, False, 0),
        "eager": (True, False, True, False, 0),
        "capture": (True, True, True, False, 0),
        "replay": (True, True, True, False, 1),
        "graph_not_required": (True, True, False, False, 0),
    }
    applicable, eager, required, passed, count = states[failure]
    return GraphShapeEvidence(
        descriptor,
        applicable,
        eager,
        required,
        count if replays is None else replays,
        passed,
        failure,
    )


def _raw(
    requirement: GraphVerificationRequirement,
    *,
    failure: tuple[str, str] | None = None,
    replay_override: tuple[str, int] | None = None,
    domain_incomplete: str | None = None,
    not_applicable: str | None = None,
) -> GraphVerificationRawEvidence:
    by_member: dict[str, list[GraphVariantEvidence]] = {}
    for expected in requirement.variants:
        shapes = []
        for descriptor in expected.shape_descriptor_digests:
            kind = "not_applicable" if expected.slot_id == not_applicable else "none"
            if failure == (expected.slot_id, descriptor):
                kind = "capture"
            count = (
                replay_override[1]
                if replay_override is not None
                and replay_override[0] == descriptor
                else None
            )
            shapes.append(_shape(descriptor, kind, replays=count))
        by_member.setdefault(expected.slot_id, []).append(
            GraphVariantEvidence(
                expected.slot_id,
                expected.variant_id,
                expected.slot_id != not_applicable,
                expected.slot_id != domain_incomplete,
                tuple(shapes),
            )
        )
    return GraphVerificationRawEvidence(
        requirement.digest,
        tuple(
            GraphMemberEvidence(slot, tuple(variants))
            for slot, variants in sorted(by_member.items())
        ),
    )


def _ref(
    requirement: GraphVerificationRequirement,
    raw: GraphVerificationRawEvidence,
) -> GraphVerificationEvidenceRef:
    return GraphVerificationEvidenceRef(requirement.binding, requirement.digest, raw.digest)


def _grade(
    requirement: GraphVerificationRequirement,
    raw: GraphVerificationRawEvidence,
) -> GraphVerificationGrade:
    return regrade_graph_verification(requirement, _ref(requirement, raw), raw)


def test_exact_atomic_graph_evidence_round_trips_and_passes_only_the_veto():
    requirement = _requirement()
    raw = _raw(requirement)
    reference = _ref(requirement, raw)
    grade = regrade_graph_verification(requirement, reference, raw)

    assert grade.decision is QualificationDecision.PASS
    assert grade.veto_passed
    assert grade.reason == "graph_verification_pass"
    assert GraphVerificationRequirement.from_dict(requirement.to_dict()) == requirement
    assert GraphVerificationRawEvidence.from_dict(raw.to_dict()) == raw
    assert GraphVerificationEvidenceRef.from_dict(reference.to_dict()) == reference
    assert GraphVerificationGrade.from_dict(grade.to_dict()) == grade
    assert len({requirement.digest, raw.digest, reference.digest, grade.digest}) == 4
    assert "score" not in grade.to_dict() and "crown" not in grade.to_dict()


def test_qualification_import_is_stdlib_only_and_does_not_import_torch():
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import optima.eval.qualification; assert 'torch' not in sys.modules",
        ],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("which", ["both", "ref", "raw"])
def test_absent_or_unrun_graph_evidence_is_no_decision(which):
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    reference = _ref(requirement, raw)
    grade = regrade_graph_verification(
        requirement,
        None if which in {"both", "raw"} else reference,
        None if which in {"both", "ref"} else raw,
    )
    assert grade.decision is QualificationDecision.NO_DECISION
    assert grade.reason == "graph_evidence_missing"


def test_atomic_missing_member_is_incomplete_not_a_partial_pass():
    requirement = _requirement()
    complete = _raw(requirement)
    missing = replace(complete, members=complete.members[:1])
    grade = _grade(requirement, missing)
    assert grade.decision is QualificationDecision.NO_DECISION
    assert grade.reason == "graph_member_coverage_incomplete"


def test_missing_variant_or_shape_is_no_decision():
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    member = raw.members[0]
    variant = member.variants[0]

    wrong_variant = replace(variant, variant_id="other")
    variant_gap = replace(raw, members=(replace(member, variants=(wrong_variant,)),))
    assert _grade(requirement, variant_gap).reason == "graph_variant_coverage_incomplete"

    shape_gap = replace(variant, shapes=variant.shapes[:1])
    shape_raw = replace(raw, members=(replace(member, variants=(shape_gap,)),))
    assert _grade(requirement, shape_raw).reason == "graph_shape_coverage_incomplete"


@pytest.mark.parametrize("failure", ["eager", "capture", "replay", "graph_not_required"])
def test_bound_negative_graph_proof_is_fail(failure):
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    member, variant = raw.members[0], raw.members[0].variants[0]
    broken = _shape(variant.shapes[0].descriptor_digest, failure)
    raw = replace(
        raw,
        members=(
            replace(
                member,
                variants=(replace(variant, shapes=(broken, variant.shapes[1])),),
            ),
        ),
    )
    grade = _grade(requirement, raw)
    assert grade.decision is QualificationDecision.FAIL
    assert failure in grade.reason


def test_eager_correct_capture_wrong_is_not_rescued_by_other_shapes():
    requirement = _requirement()
    slot = requirement.binding.members[0].slot_id
    descriptor = requirement.variants[0].shape_descriptor_digests[0]
    grade = _grade(requirement, _raw(requirement, failure=(slot, descriptor)))
    assert grade.decision is QualificationDecision.FAIL
    assert grade.reason == "graph_capture_failed"


def test_domain_incomplete_and_entirely_off_domain_members_fail():
    requirement = _requirement()
    slot = requirement.binding.members[0].slot_id
    domain = _grade(requirement, _raw(requirement, domain_incomplete=slot))
    outside = _grade(requirement, _raw(requirement, not_applicable=slot))
    assert (domain.decision, domain.reason) == (
        QualificationDecision.FAIL,
        "graph_domain_coverage_failed",
    )
    assert (outside.decision, outside.reason) == (
        QualificationDecision.FAIL,
        "graph_applicability_failed",
    )


def test_required_shape_cannot_be_relabelled_not_applicable():
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    member, variant = raw.members[0], raw.members[0].variants[0]
    hidden = _shape(variant.shapes[0].descriptor_digest, "not_applicable")
    raw = replace(
        raw,
        members=(
            replace(
                member,
                variants=(replace(variant, shapes=(hidden, variant.shapes[1])),),
            ),
        ),
    )
    grade = _grade(requirement, raw)
    assert (grade.decision, grade.reason) == (
        QualificationDecision.FAIL,
        "graph_applicability_failed",
    )


def test_success_with_wrong_replay_count_is_unrun_policy_no_decision():
    requirement = _requirement(atomic=False)
    descriptor = requirement.variants[0].shape_descriptor_digests[0]
    grade = _grade(requirement, _raw(requirement, replay_override=(descriptor, 2)))
    assert grade.decision is QualificationDecision.NO_DECISION
    assert grade.reason == "graph_replay_count_mismatch"


def test_swapped_launch_or_tampered_raw_evidence_is_no_decision():
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    reference = _ref(requirement, raw)
    swapped_binding = replace(
        requirement.binding, candidate_launch_digest=_d("another-launch")
    )
    swapped = replace(requirement, binding=swapped_binding)
    grade = regrade_graph_verification(swapped, reference, raw)
    assert (grade.decision, grade.reason) == (
        QualificationDecision.NO_DECISION,
        "graph_identity_mismatch",
    )

    tampered_ref = replace(reference, raw_evidence_digest=_d("forged-raw"))
    grade = regrade_graph_verification(requirement, tampered_ref, raw)
    assert (grade.decision, grade.reason) == (
        QualificationDecision.NO_DECISION,
        "graph_evidence_tampered",
    )


def test_strict_canonical_records_reject_duplicates_reorder_floats_and_audit_fields():
    requirement = _requirement()
    raw = _raw(requirement)

    with pytest.raises(QualificationError, match="ordered"):
        replace(requirement.binding, members=tuple(reversed(requirement.binding.members)))
    with pytest.raises(QualificationError, match="duplicates"):
        replace(
            requirement.binding,
            members=(requirement.binding.members[0], requirement.binding.members[0]),
        )
    with pytest.raises(QualificationError, match="ordered"):
        replace(requirement, variants=tuple(reversed(requirement.variants)))
    with pytest.raises(QualificationError, match="integer"):
        GraphVerificationRequirement.from_dict(
            {**requirement.to_dict(), "expected_graph_replays": 3.0}
        )

    untrusted = raw.to_dict()
    untrusted["audit_passed"] = True
    with pytest.raises(QualificationError, match="fields"):
        GraphVerificationRawEvidence.from_dict(untrusted)
    untrusted = raw.members[0].variants[0].shapes[0].to_dict()
    untrusted["fully_verified"] = True
    with pytest.raises(QualificationError, match="fields"):
        GraphShapeEvidence.from_dict(untrusted)


def test_inconsistent_shape_claims_and_reordered_raw_rows_reject_at_parse_time():
    requirement = _requirement()
    raw = _raw(requirement)
    descriptor = requirement.variants[0].shape_descriptor_digests[0]
    with pytest.raises(QualificationError, match="inconsistent"):
        GraphShapeEvidence(descriptor, True, True, True, 3, False, "none")
    with pytest.raises(QualificationError, match="ordered"):
        replace(raw, members=tuple(reversed(raw.members)))


def test_reference_profile_and_precommitted_selection_round_trip():
    reference = _reference()
    requirement = _requirement(atomic=False)
    profile = QualificationProfile(
        reference,
        _d("calibration-context"),
        _d("calibration"),
        requirement.digest,
        ("mean_nll", "task_score", "topk_kl"),
        "2",
        10,
        2,
        2,
        _d("support-policy"),
        _d("hidden-task-policy"),
        _d("runtime-resource-policy"),
        True,
        2,
    )
    assert ReferenceManifest.from_dict(reference.to_dict()) == reference
    assert QualificationProfile.from_dict(profile.to_dict()) == profile

    prompts = tuple(sorted(_d(f"prompt:{index}") for index in range(8)))
    secret = b"pre-result secret" * 4
    commitment = SelectionCommitment.seal(
        source_plan_digest=_d("cohort"),
        reference_manifest=reference,
        entropy_source_digest=_d("future-block-source"),
        prompt_digests=prompts,
        select_count=3,
        secret=secret,
    )
    entropy = SelectionEntropyReceipt(
        commitment.entropy_source_digest,
        commitment.digest,
        _d("future-block-value"),
        _d("future-block-receipt"),
    )
    receipt = SelectionReceipt.reveal(
        commitment,
        secret=secret,
        entropy=entropy,
        sealed_cohort_trajectory_digest=_d("sealed-trajectories"),
    )
    assert SelectionCommitment.from_dict(commitment.to_dict()) == commitment
    assert SelectionReceipt.from_dict(receipt.to_dict()).reopen(commitment, entropy) == receipt
    assert len(receipt.selected_prompt_digests) == 3
    rebound = SelectionReceipt.reveal(
        commitment,
        secret=secret,
        entropy=entropy,
        sealed_cohort_trajectory_digest=_d("different-sealed-trajectories"),
    )
    assert rebound.selected_prompt_digests == receipt.selected_prompt_digests


def test_selection_rejects_late_substitution_or_forged_result():
    reference = _reference()
    prompts = tuple(sorted(_d(f"prompt:{index}") for index in range(4)))
    commitment = SelectionCommitment.seal(
        source_plan_digest=_d("cohort"),
        reference_manifest=reference,
        entropy_source_digest=_d("entropy-source"),
        prompt_digests=prompts,
        select_count=2,
        secret=b"a" * 32,
    )
    with pytest.raises(QualificationError, match="does not open"):
        entropy = SelectionEntropyReceipt(
            commitment.entropy_source_digest,
            commitment.digest,
            _d("entropy"),
            _d("entropy-receipt"),
        )
        SelectionReceipt.reveal(
            commitment,
            secret=b"b" * 32,
            entropy=entropy,
            sealed_cohort_trajectory_digest=_d("trajectories"),
        )
    receipt = SelectionReceipt.reveal(
        commitment,
        secret=b"a" * 32,
        entropy=entropy,
        sealed_cohort_trajectory_digest=_d("trajectories"),
    )
    wrong = tuple(sorted(set(prompts) - set(receipt.selected_prompt_digests)))
    with pytest.raises(QualificationError, match="does not reproduce"):
        replace(receipt, selected_prompt_digests=wrong).reopen(commitment, entropy)


def _publish_graph_bytes(root: Path, payload: bytes):
    return publish_evidence(
        root,
        payload,
        domain=GRAPH_EVIDENCE_DOMAIN,
        media_type=GRAPH_EVIDENCE_MEDIA_TYPE,
        schema=GRAPH_EVIDENCE_SCHEMA,
    )


def test_controller_store_graph_artifact_reopens_and_regrades(tmp_path: Path):
    requirement = _requirement()
    raw = _raw(requirement)
    evidence_ref = _ref(requirement, raw)
    artifact = _publish_graph_bytes(tmp_path / "evidence", canonical_json_bytes(raw.to_dict()))

    grade = reopen_graph_verification(
        tmp_path / "evidence", artifact, requirement, evidence_ref
    )
    assert grade == regrade_graph_verification(requirement, evidence_ref, raw)
    assert grade.decision is QualificationDecision.PASS


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("domain", "qualification.other"),
        ("media_type", "application/json"),
        ("schema", "optima.qualification.other.v1"),
    ),
)
def test_graph_artifact_requires_exact_type_and_metadata(tmp_path: Path, field, value):
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    artifact = _publish_graph_bytes(tmp_path / "evidence", canonical_json_bytes(raw.to_dict()))
    with pytest.raises(QualificationError, match="artifact reference"):
        reopen_graph_verification(
            tmp_path / "evidence",
            replace(artifact, **{field: value}),
            requirement,
            _ref(requirement, raw),
        )
    with pytest.raises(QualificationError, match="artifact reference"):
        reopen_graph_verification(
            tmp_path / "evidence", artifact.to_dict(), requirement, _ref(requirement, raw)
        )


def test_graph_artifact_rejects_noncanonical_or_malformed_json(tmp_path: Path):
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    canonical = canonical_json_bytes(raw.to_dict())
    duplicate = (
        b'{"members":[],"members":[],"policy_version":"graph-verification-veto.v1",'
        b'"requirement_digest":"' + requirement.digest.encode() +
        b'","schema_version":1}'
    )
    attacks = (
        duplicate,
        b'{"value":NaN}',
        canonical + b"{}",
        b" " + canonical,
        json.dumps(raw.to_dict()).encode(),
        b"\xff",
    )
    for index, payload in enumerate(attacks):
        root = tmp_path / f"evidence-{index}"
        artifact = _publish_graph_bytes(root, payload)
        with pytest.raises(QualificationError, match="JSON"):
            reopen_graph_verification(root, artifact, requirement, _ref(requirement, raw))


def test_graph_artifact_cannot_substitute_parsed_grade_or_other_raw(tmp_path: Path):
    requirement = _requirement(atomic=False)
    raw = _raw(requirement)
    evidence_ref = _ref(requirement, raw)

    grade_bytes = canonical_json_bytes(_grade(requirement, raw).to_dict())
    grade_root = tmp_path / "grade"
    with pytest.raises(QualificationError, match="fields"):
        reopen_graph_verification(
            grade_root,
            _publish_graph_bytes(grade_root, grade_bytes),
            requirement,
            evidence_ref,
        )

    altered = _raw(
        requirement,
        failure=(
            requirement.variants[0].slot_id,
            requirement.variants[0].shape_descriptor_digests[0],
        ),
    )
    altered_root = tmp_path / "altered"
    result = reopen_graph_verification(
        altered_root,
        _publish_graph_bytes(altered_root, canonical_json_bytes(altered.to_dict())),
        requirement,
        evidence_ref,
    )
    assert (result.decision, result.reason) == (
        QualificationDecision.NO_DECISION,
        "graph_evidence_tampered",
    )


def test_lifecycle_derives_prompt_pool_and_exact_selected_trajectories(tmp_path: Path):
    from tests.test_scoring import _lifecycle

    lifecycle, delta, _case, _calibration, _runtime_policy = _lifecycle(tmp_path)
    prompts = lifecycle_prompt_digests(lifecycle)
    assert len(prompts) == 3
    cohort = cohort_trajectory_digest(lifecycle)
    selected = selected_trajectory_digest(
        lifecycle, selected_delta_digest=delta, selected_prompt_digests=prompts[:2]
    )
    assert len({cohort, selected}) == 2

    row = lifecycle.candidates[0]
    session = row.execution.session
    batches = list(session.batches)
    evidence = batches[1].evidence
    prompt = evidence.prompts[0]
    corrupted = replace(prompt, output_ids=(999,) + prompt.output_ids[1:])
    batches[1] = replace(
        batches[1], evidence=replace(evidence, prompts=(corrupted,))
    )
    changed = replace(
        lifecycle,
        candidates=(replace(row, execution=replace(
            row.execution, session=replace(session, batches=tuple(batches))
        )),),
    )
    assert cohort_trajectory_digest(changed) != cohort
    assert selected_trajectory_digest(
        changed, selected_delta_digest=delta, selected_prompt_digests=prompts[:2]
    ) != selected


def test_trajectory_projection_rejects_subset_relabel_and_short_topk(tmp_path: Path):
    from tests.test_scoring import _lifecycle

    lifecycle, delta, _case, _calibration, _runtime_policy = _lifecycle(tmp_path)
    with pytest.raises(QualificationError, match="prompts differ"):
        selected_trajectory_digest(
            lifecycle,
            selected_delta_digest=delta,
            selected_prompt_digests=(_d("not-a-live-prompt"),),
        )
    execution = lifecycle.candidates[0].execution
    batches = list(execution.session.batches)
    prompt = batches[0].evidence.prompts[0]
    batches[0] = replace(
        batches[0], evidence=replace(
            batches[0].evidence,
            prompts=(replace(prompt, top_logprobs=prompt.top_logprobs[:-1]),),
        ),
    )
    broken = replace(
        lifecycle,
        candidates=(replace(
            lifecycle.candidates[0],
            execution=replace(execution, session=replace(execution.session, batches=tuple(batches))),
        ),),
    )
    with pytest.raises(QualificationError, match="coverage"):
        cohort_trajectory_digest(broken)


def test_quality_binding_projects_exact_lifecycle_coverage(tmp_path: Path):
    from optima.eval.reference_quality import ReferenceQualityRawBinding
    from tests.test_scoring import _lifecycle

    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    reference = ReferenceManifest(
        *(_d(f"pristine:{index}") for index in range(3)),
        case.launch.runtime_digest, case.launch.base_engine_digest, case.launch.arena_digest,
        lifecycle.candidates[0].arm.candidate.catalog_digest,
        case.launch.controller_distribution_digest, case.launch.worker_distribution_digest,
        case.launch.model_revision_digest, case.launch.model_manifest_digest,
        case.launch.model_content_digest, case.launch.hardware.digest,
        calibration.context.workload_digest, _d("tokenizer"), _d("hidden-corpus"),
        _d("hidden-judge"), _d("entropy-source"),
    )
    profile = QualificationProfile(
        reference, calibration.context.digest, calibration.digest, _d("graph-requirement"),
        tuple(row.name for row in calibration.quality_metrics), "2", 10, 1, 2,
        _d("support-policy"), _d("hidden-task-policy"), runtime_policy, True, 2,
    )
    prompts = lifecycle_prompt_digests(lifecycle)
    commitment = SelectionCommitment.seal(
        source_plan_digest=lifecycle.source.digest, reference_manifest=reference,
        entropy_source_digest=reference.selection_policy_digest,
        prompt_digests=prompts, select_count=2, secret=b"s" * 32,
    )
    entropy = SelectionEntropyReceipt(
        commitment.entropy_source_digest, commitment.digest,
        _d("entropy-value"), _d("entropy-authority"),
    )
    selection = SelectionReceipt.reveal(
        commitment, secret=b"s" * 32, entropy=entropy,
        sealed_cohort_trajectory_digest=cohort_trajectory_digest(lifecycle),
    )
    binding = ReferenceQualityRawBinding(
        _d("qualification"), reference.digest, calibration.digest, selection.digest,
        _d("lifecycle"), selected_trajectory_digest(
            lifecycle, selected_delta_digest=delta,
            selected_prompt_digests=selection.selected_prompt_digests,
        ),
        selected_trajectory_projection_digest(
            lifecycle, selected_delta_digest=delta,
            selected_prompt_digests=selection.selected_prompt_digests,
        ), selection.selected_prompt_digests,
        _d("t-session"), profile.support_policy_digest,
        derived_hidden_task_plan_digest(profile, selection.selected_prompt_digests),
        profile.nll_tail_threshold, 10, 1, 2,
    )
    assert validate_quality_binding(
        profile, binding, lifecycle, selected_delta_digest=delta,
        commitment=commitment, entropy=entropy, selection=selection,
    ) == binding
    with pytest.raises(QualificationError, match="frozen workload"):
        validate_quality_binding(
            replace(profile, hidden_task_policy_digest=_d("other-hidden-policy")),
            binding, lifecycle, selected_delta_digest=delta,
            commitment=commitment, entropy=entropy, selection=selection,
        )
    with pytest.raises(QualificationError, match="frozen workload"):
        validate_quality_binding(
            profile, replace(binding, tokens_per_prompt=1), lifecycle,
            selected_delta_digest=delta, commitment=commitment,
            entropy=entropy, selection=selection,
        )
