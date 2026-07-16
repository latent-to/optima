"""Unit tests for the noise-robust speedup scorer (optima/eval/scoring.py).

The whole point of this module is to make a sub-10% real win resolvable on a box
whose clocks can't be locked, and to refuse to crown on measurement noise. These
tests pin both halves: a genuine win passes, and noise alone never does.
"""

from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from optima.eval.calibration import CalibrationContext, SpeedCalibration
from optima.eval.device_state import (
    DeviceStateActiveReceipt,
    DeviceStateReceipt,
    DeviceStateSample,
)
from optima.eval.marginal_runtime import (
    CandidateLifecycleEvidence,
    MarginalLifecycleEvidence,
    MarginalRuntimeError,
    run_marginal_lifecycle,
)
from optima.eval.native_artifact import publish_native_artifact
from optima.eval.oci_backend import EngineExecutionEvidence, runtime_identity_from_preflight
from optima.eval.oci_prebuild import OCIPrebuildResult
from optima.eval.oci_outer_session import (
    BatchExecutionEvidence,
    OuterSessionInfrastructureError,
    SessionExecutionEvidence,
    SessionExecutionPlan,
    require_decode_dominant_plan,
)
from optima.eval.oci_session_protocol import (
    BatchEvidence,
    PromptEvidence,
)
from optima.eval.scoring import (
    RawSpeedEvidenceError,
    marginal_workload_digest,
    project_marginal_speed,
    recompute_charged_rate,
    relative_spread,
    score_speedup,
)
from tests.test_calibration import _manifest as calibration_manifest
from tests.test_marginal_runtime import _case as runtime_case
from tests.test_marginal_runtime import _prepared as prepared_runtime


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _binding(label: str) -> str:
    return _digest(label)[:32]


def _session(plan: SessionExecutionPlan, *, label: str, scale: float) -> SessionExecutionEvidence:
    times = ((0.1, 0.3), (0.4, 0.6), (0.7, 0.9))
    batches = []
    for index, (started, completed) in enumerate(times):
        prompts = tuple(
            PromptEvidence(
                tuple(range(plan.max_new_tokens)),
                tuple(((0.0, 0),) for _ in range(plan.max_new_tokens)),
            )
            for _ in plan.prompt_batches[index]
        )
        tokens = len(prompts) * plan.max_new_tokens
        batches.append(BatchExecutionEvidence(
            index,
            _binding(f"request:{label}:{index}"),
            _binding(f"nonce:{label}:{index}"),
            started * scale,
            completed * scale,
            tokens,
            BatchEvidence(prompts),
        ))
    return SessionExecutionEvidence(
        _binding("session:" + label),
        plan.launch_digest,
        plan.expected_preflight,
        0.0,
        tuple(batches),
        plan.warmup_count,
        plan.conditioning_count,
        0.0,
        0.6 * scale,
        batches[0].token_numerator + batches[1].token_numerator,
        1.0 * scale,
    )


class _TypedExecutor:
    def __init__(self, root: Path, scales: tuple[float, ...], runtime_policy: str) -> None:
        self.root = root
        self.scales = scales
        self.runtime_policy = runtime_policy
        self.index = 0
        (root / "publications").mkdir(parents=True)

    def _publication(self, build_digest: str):
        source = self.root / "native" / build_digest
        source.mkdir(parents=True, exist_ok=True)
        (source / "kernel.bin").write_bytes(b"typed test artifact")
        return publish_native_artifact(
            source, self.root / "publications", build_spec_digest=build_digest
        )

    def _devices(self, session: SessionExecutionEvidence, launch_id: str, sequence: int):
        ids, config = (0,), _digest("device-configuration")
        policy = self.device_policy
        first_timed = session.batches[session.warmup_count].request_started_at
        idle = DeviceStateSample(-1.5, (), (), True, "idle")
        active_sample = DeviceStateSample(0.02, (), (), False, "active", True, "ok")
        ready_sample = DeviceStateSample(0.03, (), (), False, "ready", True, "ok")
        post_sample = DeviceStateSample(session.session_completed_at + 0.02, (), (), True, "idle")
        return (
            DeviceStateReceipt("optima.device-state-receipt.v1", sequence, launch_id, "pre", ids, config, policy, -2.0, -1.0, 1, (idle,)),
            DeviceStateActiveReceipt("optima.device-state-active-receipt.v2", sequence + 1, launch_id, "final-warmup", ids, config, policy, 0.01, first_timed - 0.01, 1, 1, 1, (active_sample, ready_sample)),
            DeviceStateReceipt("optima.device-state-receipt.v1", sequence + 2, launch_id, "post", ids, config, policy, session.session_completed_at + 0.01, session.session_completed_at + 0.03, 1, (post_sample,)),
        )

    def execute(self, launch, binding, mount, plan, *, deadline):
        del deadline
        index, self.index = self.index, self.index + 1
        label = f"arm-{index}"
        session = _session(plan, label=label, scale=self.scales[index])
        publication = self._publication(binding.native_build_spec.digest)
        prebuild = OCIPrebuildResult(
            launch.digest, binding.native_build_spec.digest, publication, None, None
        )
        self.device_policy = launch.hardware.device_policy_digest
        receipts = self._devices(session, "runtime-" + _binding(label), index * 3 + 1)
        receipt = binding.runtime_preflight_receipt
        return EngineExecutionEvidence(
            "optima.oci-engine-execution.v1",
            launch.digest,
            runtime_identity_from_preflight(receipt),
            receipt.sha256,
            mount.digest,
            self.runtime_policy,
            prebuild,
            publication.publication_digest,
            _digest("argv:" + label),
            (),
            receipts,
            session,
        )


def _lifecycle(tmp_path: Path, *, after_scale: float = 1.002):
    case = runtime_case(tmp_path)
    case.session = replace(
        case.session,
        prompt_batches=(("warmup",), ("timed-1",), ("timed-2",)),
        max_new_tokens=10,
        top_logprobs_num=1,
    )
    prepared = prepared_runtime(case)
    runtime_policy = _digest("runtime-resource-policy")
    lifecycle = run_marginal_lifecycle(
        prepared,
        executor=_TypedExecutor(tmp_path / "executor", (1.0, 0.75, after_scale), runtime_policy),
        model_mount=case.mount,
        deadline=10_000.0,
    )
    context = CalibrationContext(
        _digest("reference-manifest"),
        case.launch.arena_digest,
        case.launch.runtime_digest,
        case.launch.base_engine_digest,
        case.launch.model_revision_digest,
        case.launch.model_manifest_digest,
        case.launch.model_content_digest,
        case.launch.hardware.digest,
        marginal_workload_digest(prepared.baseline_session_plan),
        _digest("verification-policy"),
        case.launch.controller_distribution_digest,
    )
    calibration = replace(
        calibration_manifest(), context=context, speed=SpeedCalibration("0.02", "2", "0.1")
    )
    return lifecycle, case.arm.selected_delta_digest, case, calibration, runtime_policy


def _project(lifecycle, delta, case, calibration, runtime_policy):
    return project_marginal_speed(
        lifecycle,
        selected_delta_digest=delta,
        calibration=calibration,
        expected_context=calibration.context,
        model_mount=case.mount,
        expected_launch_resource_policy_digest=case.launch.resource_policy_digest,
        expected_runtime_resource_policy_digest=runtime_policy,
        expected_device_policy_digest=case.launch.hardware.device_policy_digest,
    )


def _replace_candidate_session(
    lifecycle: MarginalLifecycleEvidence, session: SessionExecutionEvidence
) -> MarginalLifecycleEvidence:
    row = lifecycle.candidates[0]
    receipts = list(row.execution.device_receipts)
    post = receipts[2]
    if session.session_completed_at >= post.started_monotonic_s:
        sample = replace(post.samples[0], monotonic_s=session.session_completed_at + 0.02)
        receipts[2] = replace(
            post,
            started_monotonic_s=session.session_completed_at + 0.01,
            completed_monotonic_s=session.session_completed_at + 0.03,
            samples=(sample,),
        )
    execution = replace(row.execution, session=session, device_receipts=tuple(receipts))
    return MarginalLifecycleEvidence(
        lifecycle.prepared,
        lifecycle.baseline_before,
        (CandidateLifecycleEvidence(row.candidate, execution),),
        lifecycle.baseline_after,
    )


def test_relative_spread_two_reads_is_range_over_mean():
    # The default bookend has exactly 2 baseline reads; spread is the honest gap.
    assert relative_spread([100.0, 110.0]) == (10.0 / 105.0)


def test_relative_spread_unmeasurable_below_two():
    assert relative_spread([100.0]) == float("inf")
    assert relative_spread([]) == float("inf")


def test_genuine_win_on_stable_box_passes():
    # Baselines agree (1% spread), candidate is a clean 12% faster -> real win.
    v = score_speedup([100.0, 101.0], 113.0, min_margin=0.02, k=2.0, max_noise=0.10)
    assert v.confident
    assert v.passed_speedup
    assert v.speedup > 1.11


def test_noise_alone_does_not_crown():
    # No real improvement (candidate ~= baseline mean) but candidate happens to read
    # a hair high; the noise-derived bar must reject it.
    v = score_speedup([100.0, 108.0], 106.0, min_margin=0.02, k=2.0, max_noise=0.10)
    # baseline spread = 8/104 ~= 7.7% -> required ~= 1 + 2*0.077 = 1.154; speedup ~1.019.
    assert not v.passed_speedup
    assert v.required > 1.15


def test_too_noisy_is_no_decision_not_a_pass():
    # Bracketing baselines disagree by >max_noise: untrustworthy round, never crown,
    # even if the raw ratio looks huge.
    v = score_speedup([100.0, 140.0], 150.0, max_noise=0.10)
    assert not v.confident
    assert not v.passed_speedup
    assert "NO-DECISION" in v.detail


def test_single_baseline_cannot_be_confident():
    # The legacy 2-launch shape (one baseline) can't measure noise -> not crownable.
    v = score_speedup([100.0], 130.0)
    assert not v.confident
    assert not v.passed_speedup
    assert "single baseline" in v.detail


def test_min_margin_floor_applies_on_a_perfectly_stable_box():
    # Zero measured noise still requires clearing the floor margin.
    v = score_speedup([100.0, 100.0], 101.0, min_margin=0.02, k=2.0)
    assert v.noise == 0.0
    assert v.required == 1.02
    assert not v.passed_speedup  # 1.01 < 1.02
    v2 = score_speedup([100.0, 100.0], 103.0, min_margin=0.02, k=2.0)
    assert v2.passed_speedup  # 1.03 >= 1.02 on a stable box


def test_a_real_loss_is_a_loss_not_no_decision():
    v = score_speedup([100.0, 101.0], 90.0, max_noise=0.10)
    assert v.confident  # the box was stable; we trust the verdict
    assert not v.passed_speedup
    assert v.speedup < 1.0


def test_multi_candidate_reads_score_on_the_mean():
    # B C B' C' B'' shape: two candidate reads average before the ratio.
    v = score_speedup([100.0, 101.0, 99.0], [113.0, 111.0], min_margin=0.02, k=2.0, max_noise=0.10)
    assert v.n_candidates == 2
    assert v.confident
    assert v.passed_speedup
    assert abs(v.speedup - (112.0 / 100.0)) < 1e-9


def test_single_candidate_read_keeps_legacy_verdict():
    # The historical B/C/B' shape must be bit-identical through the new path.
    legacy = score_speedup([100.0, 108.0], 106.0, min_margin=0.02, k=2.0, max_noise=0.10)
    wrapped = score_speedup([100.0, 108.0], [106.0], min_margin=0.02, k=2.0, max_noise=0.10)
    assert legacy.n_candidates == wrapped.n_candidates == 1
    assert (legacy.speedup, legacy.noise, legacy.required, legacy.passed_speedup,
            legacy.confident, legacy.detail) == (
        wrapped.speedup, wrapped.noise, wrapped.required, wrapped.passed_speedup,
        wrapped.confident, wrapped.detail)


def test_noisy_candidate_reads_are_no_decision():
    # 2026-07-16 forensics: two honest candidate legs spread 7.2% on a boot draw.
    # With tight baselines, that spread alone must block the crown at max_noise 5%.
    v = score_speedup([100.0, 100.5], [107.2, 100.0], max_noise=0.05)
    assert not v.confident
    assert not v.passed_speedup
    assert "candidate drift" in v.detail


def test_candidate_spread_raises_the_required_bar():
    # Within the noise ceiling, a spread candidate raises the bar exactly like a
    # spread baseline: noise = max(baseline, candidate) feeds 1 + k*noise.
    v = score_speedup([100.0, 100.0], [104.0, 100.0], min_margin=0.005, k=2.0, max_noise=0.10)
    assert v.confident
    assert abs(v.noise - (4.0 / 102.0)) < 1e-9
    assert abs(v.required - (1.0 + 2.0 * 4.0 / 102.0)) < 1e-9
    assert not v.passed_speedup  # mean 102 -> 1.020 < required ~1.078


@pytest.mark.parametrize(
    ("baselines", "candidate"),
    (
        ([True, 100.0], 110.0),
        ([0.0, 100.0], 110.0),
        ([-1.0, 100.0], 110.0),
        ([float("nan"), 100.0], 110.0),
        ([float("inf"), 100.0], 110.0),
        ([100.0, 101.0], False),
        ([100.0, 101.0], 0.0),
        ([100.0, 101.0], float("inf")),
        ([100.0, 101.0], [110.0, 0.0]),
        ([100.0, 101.0], [110.0, float("nan")]),
        ([100.0, 101.0], [110.0, True]),
    ),
)
def test_speed_samples_fail_closed_without_filtering(baselines, candidate):
    with pytest.raises(RawSpeedEvidenceError):
        score_speedup(baselines, candidate)


@pytest.mark.parametrize(
    "policy",
    (
        {"min_margin": 0.0},
        {"min_margin": 1.0},
        {"min_margin": True},
        {"k": 0.0},
        {"k": float("inf")},
        {"max_noise": -0.1},
        {"max_noise": 1.0},
        {"max_noise": float("nan")},
    ),
)
def test_speed_policy_fails_closed(policy):
    with pytest.raises(RawSpeedEvidenceError):
        score_speedup([100.0, 101.0], 110.0, **policy)


def test_raw_projection_counts_first_timed_batch_once(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    projection = _project(lifecycle, delta, case, calibration, runtime_policy)
    # Conditioning is ready->first timed request (0.4), not the stored interval
    # ending after that request (0.6). Both timed rows are then charged once.
    assert projection.baseline_before.conditioning_seconds == pytest.approx(0.4)
    assert projection.baseline_before.timed_seconds == pytest.approx(0.5)
    assert projection.baseline_before.charged_seconds == pytest.approx(0.9)
    assert projection.baseline_before.conditioning_tokens == 10
    assert projection.baseline_before.timed_tokens == 20
    assert projection.baseline_before.charged_tokens == 30
    assert projection.baseline_before.tokens_per_second == pytest.approx(30 / 0.9)
    assert projection.candidate.tokens_per_second == pytest.approx(30 / 0.675)
    assert projection.verdict.confident and projection.verdict.passed_speedup
    assert len(projection.evidence_digest) == 64
    assert projection.evidence_digest == _project(
        lifecycle, delta, case, calibration, runtime_policy
    ).evidence_digest


def test_projection_thresholds_come_only_from_frozen_calibration(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    strict = replace(calibration, speed=SpeedCalibration("0.5", "2", "0.1"))
    projection = _project(lifecycle, delta, case, strict, runtime_policy)
    assert projection.verdict.required == 1.5
    assert not projection.verdict.passed_speedup


def test_projection_rejects_stale_context_and_provisional_policy(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    stale = replace(
        calibration,
        context=replace(calibration.context, workload_digest=_digest("stale-workload")),
    )
    with pytest.raises(RawSpeedEvidenceError, match="context differs"):
        _project(lifecycle, delta, case, stale, runtime_policy)
    with pytest.raises(RawSpeedEvidenceError, match="not frozen"):
        _project(
            lifecycle,
            delta,
            case,
            replace(calibration, status="provisional"),
            runtime_policy,
        )


@pytest.mark.parametrize("field", ("runtime_identity", "prebuild", "device_receipts"))
def test_projection_rejects_untyped_or_zero_device_execution(tmp_path, field):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    replacement = () if field == "device_receipts" else SimpleNamespace()
    forged_before = replace(lifecycle.baseline_before, **{field: replacement})
    forged = MarginalLifecycleEvidence(
        lifecycle.prepared,
        forged_before,
        lifecycle.candidates,
        lifecycle.baseline_after,
    )
    with pytest.raises(RawSpeedEvidenceError):
        _project(forged, delta, case, calibration, runtime_policy)


def test_projection_binds_launch_and_runtime_resource_policies_separately(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    assert runtime_policy != case.launch.resource_policy_digest
    assert _project(lifecycle, delta, case, calibration, runtime_policy).verdict.confident
    with pytest.raises(RawSpeedEvidenceError, match="resource policy"):
        _project(lifecycle, delta, case, calibration, _digest("other-runtime-policy"))


@pytest.mark.parametrize(
    ("field", "value"),
    (("conditioning_token_numerator", 999), ("first_timed_completed_at", 0.7)),
)
def test_raw_projection_rejects_forged_session_summaries(tmp_path, field, value):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    session = replace(lifecycle.candidates[0].execution.session, **{field: value})
    forged = _replace_candidate_session(lifecycle, session)
    with pytest.raises(RawSpeedEvidenceError, match="conditioning"):
        _project(forged, delta, case, calibration, runtime_policy)


def test_raw_projection_uses_bookend_drift_for_no_decision(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(
        tmp_path, after_scale=1.30
    )
    projection = _project(lifecycle, delta, case, calibration, runtime_policy)
    assert not projection.verdict.confident
    assert not projection.verdict.passed_speedup
    assert "NO-DECISION" in projection.verdict.detail


def test_raw_projection_requires_exact_candidate_selection_and_binding(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    with pytest.raises(RawSpeedEvidenceError, match="absent or ambiguous"):
        _project(lifecycle, _digest("wrong"), case, calibration, runtime_policy)

    object.__setattr__(
        lifecycle.candidates[0].execution,
        "launch_digest",
        lifecycle.baseline_before.launch_digest,
    )
    with pytest.raises(RawSpeedEvidenceError, match="executor evidence|launch"):
        _project(lifecycle, delta, case, calibration, runtime_policy)


@pytest.mark.parametrize(
    ("started", "completed"),
    ((0.4, 0.4), (0.5, 0.4), (float("nan"), 0.6)),
)
def test_raw_projection_rejects_zero_negative_or_nonfinite_batch_clocks(
    tmp_path, started, completed
):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    session = lifecycle.candidates[0].execution.session
    rows = list(session.batches)
    rows[1] = replace(
        rows[1], request_started_at=started, response_completed_at=completed
    )
    forged = _replace_candidate_session(lifecycle, replace(session, batches=tuple(rows)))
    with pytest.raises(RawSpeedEvidenceError, match="clock|finite"):
        _project(forged, delta, case, calibration, runtime_policy)


def test_raw_projection_rejects_token_mutation(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    session = lifecycle.candidates[0].execution.session
    rows = list(session.batches)
    rows[2] = replace(rows[2], token_numerator=9)
    forged = _replace_candidate_session(lifecycle, replace(session, batches=tuple(rows)))
    with pytest.raises(RawSpeedEvidenceError, match="token evidence"):
        _project(forged, delta, case, calibration, runtime_policy)


def test_raw_projection_rejects_redistributed_prompt_tokens(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    session = lifecycle.candidates[0].execution.session
    rows = list(session.batches)
    row = rows[2]
    # Preserve the aggregate count but violate the one-prompt workload shape.
    rows[2] = replace(
        row,
        evidence=replace(
            row.evidence,
            prompts=(PromptEvidence(tuple(range(9)), ()), PromptEvidence((9,), ())),
        ),
    )
    forged = _replace_candidate_session(lifecycle, replace(session, batches=tuple(rows)))
    with pytest.raises(RawSpeedEvidenceError, match="token evidence"):
        _project(forged, delta, case, calibration, runtime_policy)


def test_raw_projection_charges_inter_request_cooldown(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    session = lifecycle.candidates[0].execution.session
    rows = list(session.batches)
    # Move the final timed request later without changing either request's own
    # elapsed duration.  A sum-of-request-times projection would miss this gap.
    rows[2] = replace(rows[2], request_started_at=1.7, response_completed_at=1.9)
    delayed = _replace_candidate_session(
        lifecycle,
        replace(session, batches=tuple(rows), session_completed_at=2.0),
    )
    projection = _project(delayed, delta, case, calibration, runtime_policy)
    assert projection.candidate.timed_seconds == pytest.approx(1.9 - 0.3)
    assert projection.candidate.tokens_per_second < projection.baseline_before.tokens_per_second


def test_projection_is_immutable_and_raw_mutation_is_revalidated(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    projection = _project(lifecycle, delta, case, calibration, runtime_policy)
    original_rate = projection.candidate.tokens_per_second
    with pytest.raises(FrozenInstanceError):
        projection.candidate.tokens_per_second = 999.0  # type: ignore[misc]

    object.__setattr__(
        lifecycle.candidates[0].execution.session.batches[1],
        "response_completed_at",
        0.4,
    )
    assert projection.candidate.tokens_per_second == original_rate
    with pytest.raises(RawSpeedEvidenceError, match="clock|conditioning"):
        _project(lifecycle, delta, case, calibration, runtime_policy)


def test_direct_session_projection_rejects_relabel_and_fake_headline(tmp_path):
    lifecycle, _delta, _case, _calibration, _runtime_policy = _lifecycle(tmp_path)
    session = lifecycle.baseline_before.session
    plan = lifecycle.prepared.baseline_session_plan
    assert recompute_charged_rate(session, plan).tokens_per_second == pytest.approx(30 / 0.9)
    with pytest.raises(RawSpeedEvidenceError, match="identity"):
        recompute_charged_rate(replace(session, launch_digest=_digest("other")), plan)
    with pytest.raises(RawSpeedEvidenceError, match="exact typed"):
        recompute_charged_rate(SimpleNamespace(tok_per_s=10**30), plan)


def test_projection_rejects_relabelled_model_mount_identity(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    wrong = replace(case.mount, arena_digest=_digest("wrong-arena"))
    relabel = lambda execution: replace(  # noqa: E731 - compact fixture transform
        execution, arena_model_receipt_digest=wrong.digest
    )
    lifecycle = replace(
        lifecycle,
        baseline_before=relabel(lifecycle.baseline_before),
        candidates=tuple(
            replace(row, execution=relabel(row.execution)) for row in lifecycle.candidates
        ),
        baseline_after=relabel(lifecycle.baseline_after),
    )
    case.mount = wrong
    with pytest.raises(RawSpeedEvidenceError, match="model/resource/device identity"):
        _project(lifecycle, delta, case, calibration, runtime_policy)


def test_projection_regrades_device_samples_and_cross_arm_order(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    receipts = list(lifecycle.baseline_before.device_receipts)
    receipts[0] = replace(
        receipts[0], samples=(replace(receipts[0].samples[-1], idle=False),)
    )
    bad_samples = replace(
        lifecycle,
        baseline_before=replace(lifecycle.baseline_before, device_receipts=tuple(receipts)),
    )
    with pytest.raises(RawSpeedEvidenceError, match="sample facts"):
        _project(bad_samples, delta, case, calibration, runtime_policy)

    receipts = tuple(
        replace(receipt, sequence=10 + index)
        for index, receipt in enumerate(lifecycle.baseline_before.device_receipts)
    )
    reordered = replace(
        lifecycle,
        baseline_before=replace(lifecycle.baseline_before, device_receipts=receipts),
    )
    with pytest.raises(RawSpeedEvidenceError, match="sequence was replayed"):
        _project(reordered, delta, case, calibration, runtime_policy)


def test_projection_rejects_overlapping_active_and_post_release_samples(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    receipts = list(lifecycle.baseline_before.device_receipts)
    active = receipts[1]
    receipts[1] = replace(
        active,
        release_sample_index=0,
        samples=(active.samples[0],),
    )
    lifecycle = replace(
        lifecycle,
        baseline_before=replace(
            lifecycle.baseline_before,
            device_receipts=tuple(receipts),
        ),
    )
    with pytest.raises(RawSpeedEvidenceError, match="sample facts"):
        _project(lifecycle, delta, case, calibration, runtime_policy)


def test_projection_accepts_ready_miss_before_authoritative_passing_tail(tmp_path):
    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    receipts = list(lifecycle.baseline_before.device_receipts)
    active = receipts[1]
    intermediate_miss = replace(
        active.samples[-1],
        monotonic_s=(
            active.samples[-2].monotonic_s + active.samples[-1].monotonic_s
        ) / 2,
        active_envelope_passed=False,
        active_envelope_reason="ready envelope not reached yet",
    )
    receipts[1] = replace(
        active,
        samples=(*active.samples[:-1], intermediate_miss, active.samples[-1]),
    )
    with_miss = replace(
        lifecycle,
        baseline_before=replace(
            lifecycle.baseline_before,
            device_receipts=tuple(receipts),
        ),
    )

    assert _project(with_miss, delta, case, calibration, runtime_policy)

    failed_tail = list(receipts)
    failed_tail[1] = replace(
        receipts[1],
        samples=(
            *receipts[1].samples[:-1],
            replace(
                receipts[1].samples[-1],
                active_envelope_passed=False,
                active_envelope_reason="ready tail failed",
            ),
        ),
    )
    with_failed_tail = replace(
        lifecycle,
        baseline_before=replace(
            lifecycle.baseline_before,
            device_receipts=tuple(failed_tail),
        ),
    )
    with pytest.raises(RawSpeedEvidenceError, match="sample facts"):
        _project(with_failed_tail, delta, case, calibration, runtime_policy)


def _lifecycle_repeat(tmp_path: Path):
    # B, C, B', C', B'' — scales make both candidate reads ~1.32x the baselines
    # with tiny within-arm spread, so the repeat-read verdict is a confident PASS.
    case = runtime_case(tmp_path)
    case.session = replace(
        case.session,
        prompt_batches=(("warmup",), ("timed-1",), ("timed-2",)),
        max_new_tokens=10,
        top_logprobs_num=1,
    )
    prepared = prepared_runtime(case)
    runtime_policy = _digest("runtime-resource-policy")
    lifecycle = run_marginal_lifecycle(
        prepared,
        executor=_TypedExecutor(
            tmp_path / "executor", (1.0, 0.75, 1.002, 0.76, 0.998), runtime_policy
        ),
        model_mount=case.mount,
        deadline=10_000.0,
        candidate_reads=2,
    )
    context = CalibrationContext(
        _digest("reference-manifest"),
        case.launch.arena_digest,
        case.launch.runtime_digest,
        case.launch.base_engine_digest,
        case.launch.model_revision_digest,
        case.launch.model_manifest_digest,
        case.launch.model_content_digest,
        case.launch.hardware.digest,
        marginal_workload_digest(prepared.baseline_session_plan),
        _digest("verification-policy"),
        case.launch.controller_distribution_digest,
    )
    calibration = replace(
        calibration_manifest(), context=context, speed=SpeedCalibration("0.02", "2", "0.1")
    )
    return lifecycle, case.arm.selected_delta_digest, case, calibration, runtime_policy


def test_repeat_read_lifecycle_projects_and_regrades_five_rates(tmp_path):
    from optima.eval.qualification_runner import SpeedWitness

    lifecycle, delta, case, calibration, runtime_policy = _lifecycle_repeat(tmp_path)
    assert len(lifecycle.candidates_repeat) == 1
    assert lifecycle.baseline_third is not None
    assert lifecycle.final_baseline is lifecycle.baseline_third

    projection = _project(lifecycle, delta, case, calibration, runtime_policy)
    assert projection.candidate_repeat is not None
    assert projection.baseline_third is not None
    assert projection.verdict.n_baselines == 3
    assert projection.verdict.n_candidates == 2
    assert projection.verdict.confident
    assert projection.verdict.passed_speedup
    assert projection.verdict.speedup > 1.25

    witness = SpeedWitness.from_projection(projection, runtime_policy)
    assert len(witness.rates) == 5
    # run order: B, C, B', C', B''
    assert witness.rates[0] == projection.baseline_before
    assert witness.rates[1] == projection.candidate
    assert witness.rates[2] == projection.baseline_after
    assert witness.rates[3] == projection.candidate_repeat
    assert witness.rates[4] == projection.baseline_third
    round_tripped = SpeedWitness.from_dict(witness.to_dict())
    assert round_tripped == witness
    grade, speedup = round_tripped.regrade(calibration, calibration.context)
    assert grade.name == "PASS"
    assert speedup == format(projection.verdict.speedup, ".17g")


def test_repeat_reads_require_the_third_baseline_bookend(tmp_path):
    lifecycle, *_ = _lifecycle_repeat(tmp_path)
    with pytest.raises(MarginalRuntimeError, match="third baseline bookend"):
        MarginalLifecycleEvidence(
            lifecycle.prepared,
            lifecycle.baseline_before,
            lifecycle.candidates,
            lifecycle.baseline_after,
            lifecycle.candidates_repeat,
            None,
        )
    with pytest.raises(MarginalRuntimeError, match="third baseline bookend"):
        MarginalLifecycleEvidence(
            lifecycle.prepared,
            lifecycle.baseline_before,
            lifecycle.candidates,
            lifecycle.baseline_after,
            (),
            lifecycle.baseline_third,
        )


def test_candidate_reads_policy_is_exact(tmp_path):
    case = runtime_case(tmp_path)
    case.session = replace(
        case.session,
        prompt_batches=(("warmup",), ("timed-1",), ("timed-2",)),
        max_new_tokens=10,
        top_logprobs_num=1,
    )
    prepared = prepared_runtime(case)
    executor = _TypedExecutor(tmp_path / "executor", (1.0,), _digest("runtime-resource-policy"))
    for reads in (0, 3, True, 2.0):
        with pytest.raises(MarginalRuntimeError, match="candidate_reads"):
            run_marginal_lifecycle(
                prepared,
                executor=executor,
                model_mount=case.mount,
                deadline=10_000.0,
                candidate_reads=reads,
            )


def test_legacy_three_leg_witness_shape_is_unchanged(tmp_path):
    from optima.eval.qualification_runner import SpeedWitness

    lifecycle, delta, case, calibration, runtime_policy = _lifecycle(tmp_path)
    projection = _project(lifecycle, delta, case, calibration, runtime_policy)
    assert projection.candidate_repeat is None
    assert projection.baseline_third is None
    assert projection.verdict.n_candidates == 1
    witness = SpeedWitness.from_projection(projection, runtime_policy)
    assert len(witness.rates) == 3
    assert SpeedWitness.from_dict(witness.to_dict()) == witness


def test_decode_dominant_plan_gate(tmp_path):
    case = runtime_case(tmp_path)
    case.session = replace(
        case.session,
        prompt_batches=(("warmup",), ("t1",), ("t2",)),
        max_new_tokens=10,
        top_logprobs_num=1,
    )
    plan = prepared_runtime(case).baseline_session_plan
    count_tokens = len  # chars-as-tokens keeps the gate arithmetic transparent
    charged = plan.prompt_batches[plan.warmup_count - plan.conditioning_count :]
    prompt_tokens = sum(len(prompt) for batch in charged for prompt in batch)
    decode_tokens = sum(len(batch) * plan.max_new_tokens for batch in charged)
    expected = decode_tokens / (decode_tokens + prompt_tokens)
    share = require_decode_dominant_plan(
        plan, count_tokens=count_tokens, min_decode_share=expected * 0.9
    )
    assert abs(share - expected) < 1e-12
    with pytest.raises(OuterSessionInfrastructureError, match="prefill-heavy"):
        require_decode_dominant_plan(plan, count_tokens=count_tokens, min_decode_share=0.99)
    with pytest.raises(OuterSessionInfrastructureError, match="min_decode_share"):
        require_decode_dominant_plan(plan, count_tokens=count_tokens, min_decode_share=1.0)
    with pytest.raises(OuterSessionInfrastructureError, match="positive ints"):
        require_decode_dominant_plan(
            plan, count_tokens=lambda prompt: 0, min_decode_share=0.5
        )
