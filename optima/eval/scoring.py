"""Recompute calibrated B/C/B-prime speed evidence from sealed lifecycle rows."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

@dataclass(frozen=True)
class SpeedupVerdict:
    speedup: float  # robust paired estimate: candidate / mean(baseline reads)
    noise: float  # measured relative spread of the baseline reads (the floor)
    required: float  # the bar it had to clear: 1 + max(min_margin, k*noise)
    passed_speedup: bool  # cleared `required` AND the round was trustworthy
    confident: bool  # False -> box too noisy this round; treat as NO-DECISION, never crown
    n_baselines: int
    detail: str = ""

class RawSpeedEvidenceError(ValueError):
    pass

@dataclass(frozen=True)
class ChargedExecutionRate:
    launch_digest: str
    session_id: str
    conditioning_tokens: int
    timed_tokens: int
    charged_tokens: int
    conditioning_seconds: float
    timed_seconds: float
    charged_seconds: float
    tokens_per_second: float

@dataclass(frozen=True)
class MarginalSpeedProjection:
    selected_delta_digest: str
    candidate_launch_digest: str
    calibration_digest: str
    calibration_context_digest: str
    workload_digest: str
    evidence_digest: str
    baseline_before: ChargedExecutionRate
    candidate: ChargedExecutionRate
    baseline_after: ChargedExecutionRate
    verdict: SpeedupVerdict


def _finite_time(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RawSpeedEvidenceError(f"{field} must be finite")
    return float(value)


def _positive_number(value: object, *, field: str) -> float:
    result = _finite_time(value, field=field)
    if result <= 0:
        raise RawSpeedEvidenceError(f"{field} must be positive")
    return result

def marginal_workload_digest(plan: object) -> str:
    from optima.eval.oci_outer_session import SessionExecutionPlan
    from optima.stack_identity import canonical_digest
    if type(plan) is not SessionExecutionPlan:
        raise RawSpeedEvidenceError("workload plan must be exact typed evidence")
    return canonical_digest(
        "optima.qualification.marginal-workload.v1",
        {
            "conditioning_count": plan.conditioning_count,
            "engine_config_digest": plan.expected_engine_config_digest,
            "max_new_tokens": plan.max_new_tokens,
            "prompt_batches": plan.prompt_batches,
            "temperature": format(plan.temperature, ".17g"),
            "top_logprobs_num": plan.top_logprobs_num,
            "warmup_count": plan.warmup_count,
        },
    )


def recompute_charged_rate(session: object, plan: object) -> ChargedExecutionRate:
    from optima.eval.oci_outer_session import (
        BatchExecutionEvidence,
        SessionExecutionEvidence,
        SessionExecutionPlan,
    )
    from optima.eval.oci_session_protocol import BatchEvidence, PromptEvidence
    if type(session) is not SessionExecutionEvidence or type(plan) is not SessionExecutionPlan:
        raise RawSpeedEvidenceError("session and plan must be exact typed evidence")
    if (session.launch_digest, session.preflight, session.warmup_count, session.conditioning_count) != (
        plan.launch_digest, plan.expected_preflight, plan.warmup_count, plan.conditioning_count
    ):
        raise RawSpeedEvidenceError("session identity or workload binding is invalid")
    counts = (plan.warmup_count, plan.conditioning_count, plan.max_new_tokens)
    if (
        any(type(value) is not int for value in counts)
        or not 1 <= plan.conditioning_count <= plan.warmup_count
        or plan.warmup_count >= len(plan.prompt_batches)
        or plan.max_new_tokens <= 0
        or type(session.batches) is not tuple
        or len(session.batches) != len(plan.prompt_batches)
    ):
        raise RawSpeedEvidenceError("session workload shape is invalid")
    ready = _finite_time(session.ready_completed_at, field="ready_completed_at")
    previous = ready
    for index, (row, prompts) in enumerate(zip(session.batches, plan.prompt_batches)):
        if type(row) is not BatchExecutionEvidence or row.batch_index != index:
            raise RawSpeedEvidenceError("batch evidence is missing, reordered, or relabeled")
        started = _finite_time(row.request_started_at, field="request_started_at")
        completed = _finite_time(row.response_completed_at, field="response_completed_at")
        if started < previous or completed <= started:
            raise RawSpeedEvidenceError("batch clock is non-monotonic or nonpositive")
        previous = completed
        expected_tokens = len(prompts) * plan.max_new_tokens
        evidence = row.evidence
        if (
            type(row.token_numerator) is not int
            or row.token_numerator != expected_tokens
            or type(evidence) is not BatchEvidence
            or type(evidence.prompts) is not tuple
            or len(evidence.prompts) != len(prompts)
            or any(type(prompt) is not PromptEvidence for prompt in evidence.prompts)
            or any(len(prompt.output_ids) != plan.max_new_tokens for prompt in evidence.prompts)
            or evidence.observed_tokens != expected_tokens
        ):
            raise RawSpeedEvidenceError("batch token evidence differs from the workload")
    first_timed = session.batches[plan.warmup_count]
    conditioning_start_index = plan.warmup_count - plan.conditioning_count
    expected_conditioning_start = (
        ready
        if conditioning_start_index == 0
        else float(session.batches[conditioning_start_index - 1].response_completed_at)
    )
    conditioning_start = _finite_time(session.conditioning_started_at, field="conditioning_started_at")
    first_timed_completed = _finite_time(session.first_timed_completed_at, field="first_timed_completed_at")
    if (
        conditioning_start != expected_conditioning_start
        or first_timed_completed != float(first_timed.response_completed_at)
    ):
        raise RawSpeedEvidenceError("conditioning boundaries do not match raw batches")
    expected_conditioning_summary = sum(row.token_numerator for row in session.batches[
        conditioning_start_index : plan.warmup_count + 1
    ])
    if session.conditioning_token_numerator != expected_conditioning_summary:
        raise RawSpeedEvidenceError("conditioning token summary differs from raw batches")
    completed = _finite_time(session.session_completed_at, field="session_completed_at")
    if completed < previous:
        raise RawSpeedEvidenceError("session completion precedes its final batch")
    conditioning_rows = session.batches[conditioning_start_index : plan.warmup_count]
    timed_rows = session.batches[plan.warmup_count :]
    conditioning_seconds = float(first_timed.request_started_at) - conditioning_start
    timed_seconds = float(timed_rows[-1].response_completed_at) - float(timed_rows[0].request_started_at)
    conditioning_tokens = sum(row.token_numerator for row in conditioning_rows)
    timed_tokens = sum(row.token_numerator for row in timed_rows)
    charged_seconds = conditioning_seconds + timed_seconds
    charged_tokens = conditioning_tokens + timed_tokens
    if not all(math.isfinite(value) and value > 0 for value in (
        conditioning_seconds, timed_seconds, charged_seconds
    )) or min(conditioning_tokens, timed_tokens, charged_tokens) <= 0:
        raise RawSpeedEvidenceError("charged time or token evidence is nonpositive")
    return ChargedExecutionRate(
        session.launch_digest, session.session_id, conditioning_tokens, timed_tokens,
        charged_tokens, conditioning_seconds, timed_seconds, charged_seconds,
        charged_tokens / charged_seconds,
    )


def _evidence_digest(value: object, *, field: str) -> str:
    from optima.stack_identity import StackIdentityError, require_sha256_hex
    try:
        result = require_sha256_hex(value, field=field)
    except StackIdentityError as exc:
        raise RawSpeedEvidenceError(str(exc)) from None
    if result == "0" * 64:
        raise RawSpeedEvidenceError(f"{field} must not be all-zero")
    return result


def _validate_speed_context(
    calibration: object, expected_context: object, launch: object, plan: object
) -> tuple[float, float, float, str]:
    from optima.eval.calibration import (
        CalibrationContext,
        CalibrationManifest,
        SpeedCalibration,
        decimal_value,
    )
    if (
        type(calibration) is not CalibrationManifest
        or type(expected_context) is not CalibrationContext
    ):
        raise RawSpeedEvidenceError("speed calibration/context is not exact typed policy")
    try:
        calibration.require_context(expected_context)
    except ValueError as exc:
        raise RawSpeedEvidenceError(str(exc)) from None
    if not calibration.thresholds_frozen or type(calibration.speed) is not SpeedCalibration:
        raise RawSpeedEvidenceError("speed calibration thresholds are not frozen")
    workload = marginal_workload_digest(plan)
    observed = CalibrationContext(
        expected_context.reference_manifest_digest,
        launch.arena_digest,
        launch.runtime_digest,
        launch.base_engine_digest,
        launch.model_revision_digest,
        launch.model_manifest_digest,
        launch.model_content_digest,
        launch.hardware.digest,
        workload,
        expected_context.verification_policy_digest,
        launch.controller_distribution_digest,
    )
    if observed != expected_context:
        raise RawSpeedEvidenceError("calibration context differs from launch or workload")
    return (
        float(decimal_value(calibration.speed.min_margin)),
        float(decimal_value(calibration.speed.noise_multiplier)),
        float(decimal_value(calibration.speed.max_noise)),
        workload,
    )

def _validate_execution_evidence(
    execution: object,
    *,
    launch: object,
    binding: object,
    model_mount: object,
    plan: object,
    expected_resource_policy_digest: str,
    expected_device_policy_digest: str,
    state: dict[str, object],
) -> ChargedExecutionRate:
    from optima.eval.device_state import DeviceStateActiveReceipt, DeviceStateReceipt, DeviceStateSample
    from optima.eval.marginal_runtime import (
        MarginalRuntimeError,
        MaterializedArmBinding,
        _require_execution,
    )
    from optima.eval.native_artifact import NativeArtifactPublication, reopen_native_artifact
    from optima.eval.oci_backend import (
        EngineExecutionEvidence,
        TrustedArenaModelMountReceipt,
        _validate_device_receipts,
    )
    from optima.eval.oci_prebuild import OCIPrebuildResult
    if (
        type(execution) is not EngineExecutionEvidence
        or type(binding) is not MaterializedArmBinding
        or type(model_mount) is not TrustedArenaModelMountReceipt
    ):
        raise RawSpeedEvidenceError("execution or materialized binding is not exact typed evidence")
    rate = recompute_charged_rate(execution.session, plan)
    local = binding.launch_binding
    prebuild = execution.prebuild
    publication = getattr(prebuild, "publication", None)
    if type(prebuild) is not OCIPrebuildResult or type(publication) is not NativeArtifactPublication:
        raise RawSpeedEvidenceError("execution prebuild/publication is not exact typed evidence")
    try:
        reopened = reopen_native_artifact(
            publication.root,
            expected_build_spec_digest=local.native_build_spec.digest,
            expected_publication_digest=publication.publication_digest,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise RawSpeedEvidenceError(f"native publication cannot be reopened: {exc}") from None
    if reopened.identity_dict() != publication.identity_dict():
        raise RawSpeedEvidenceError("reopened native publication identity differs")
    if execution.resource_policy_digest != expected_resource_policy_digest:
        raise RawSpeedEvidenceError("execution differs from the expected resource policy")
    receipts = execution.device_receipts
    if type(receipts) is not tuple or len(receipts) != 3:
        raise RawSpeedEvidenceError("execution lacks one device receipt triplet")
    pre, active, post = receipts
    if (
        type(pre) is not DeviceStateReceipt
        or type(active) is not DeviceStateActiveReceipt
        or type(post) is not DeviceStateReceipt
        or (pre.schema, active.schema, post.schema)
        != (
            "optima.device-state-receipt.v1",
            "optima.device-state-active-receipt.v2",
            "optima.device-state-receipt.v1",
        )
        or any(type(row.samples) is not tuple or not row.samples or any(type(sample) is not DeviceStateSample for sample in row.samples) for row in receipts)
        or min(pre.consecutive_idle_samples, active.consecutive_active_samples, active.post_release_ready_samples, post.consecutive_idle_samples) <= 0
        or pre.policy_sha256 != expected_device_policy_digest
        or len(pre.selected_physical_gpu_ids) != launch.hardware.visible_gpu_count
    ):
        raise RawSpeedEvidenceError("device receipts are incomplete or policy-mismatched")
    if (
        pre.consecutive_idle_samples > len(pre.samples)
        or post.consecutive_idle_samples > len(post.samples)
        or active.consecutive_active_samples > len(active.samples)
        or active.post_release_ready_samples > len(active.samples)
        or type(active.release_sample_index) is not int
        or active.release_sample_index < active.consecutive_active_samples
        or active.release_sample_index + active.post_release_ready_samples
        > len(active.samples)
        or any(not sample.idle for sample in pre.samples[-pre.consecutive_idle_samples:])
        or any(not sample.idle for sample in post.samples[-post.consecutive_idle_samples:])
        or any(
            not sample.active_envelope_passed
            for sample in active.samples[
                active.release_sample_index - active.consecutive_active_samples:
                active.release_sample_index
            ]
        )
        or any(
            not sample.active_envelope_passed
            for sample in active.samples[-active.post_release_ready_samples:]
        )
    ):
        raise RawSpeedEvidenceError("device sample facts disagree with receipt counters")
    try:
        _validate_device_receipts(receipts, launch_id=pre.launch_id)
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        raise RawSpeedEvidenceError(f"device receipt triplet is invalid: {exc}") from None
    first_timed = execution.session.batches[plan.warmup_count]
    if not (
        pre.completed_monotonic_s <= execution.session.ready_completed_at <= active.started_monotonic_s
        and active.completed_monotonic_s <= first_timed.request_started_at
        and execution.session.session_completed_at <= post.started_monotonic_s
    ):
        raise RawSpeedEvidenceError("device receipts do not bracket the retained session")
    physical = local.physical_hardware.physical_gpu_ids
    if all(value.isdecimal() for value in physical) and pre.selected_physical_gpu_ids != tuple(map(int, physical)):
        raise RawSpeedEvidenceError("device receipts name other physical GPUs")
    device_identity = (pre.selected_physical_gpu_ids, pre.configuration_sha256, pre.policy_sha256)
    if state.setdefault("device_identity", device_identity) != device_identity:
        raise RawSpeedEvidenceError("device configuration changed between arms")
    sequences = state.setdefault("sequences", set())
    current_sequences = {row.sequence for row in receipts}
    if sequences & current_sequences or (
        sequences and min(current_sequences) <= max(sequences)
    ):
        raise RawSpeedEvidenceError("device receipt sequence was replayed")
    sequences.update(current_sequences)
    try:
        _require_execution(
            execution,
            launch=launch,
            binding=local,
            mount=model_mount,
            plan=plan,
            seen_sessions=state["sessions"],
            seen_device_launches=state["device_launches"],
            seen_request_ids=state["requests"],
            seen_nonces=state["nonces"],
            seen_runtime_policies=state["runtime_policies"],
        )
    except (MarginalRuntimeError, AttributeError, TypeError, ValueError) as exc:
        raise RawSpeedEvidenceError(str(exc)) from None
    return rate


def _projection_digest(selected: str, candidate: str, calibration: str, context: str,
                       workload: str, runtime_policy: str, rates: tuple[ChargedExecutionRate, ...]) -> str:
    from optima.stack_identity import canonical_digest
    def row(rate: ChargedExecutionRate) -> list[object]:
        return [
            rate.launch_digest,
            rate.session_id,
            rate.conditioning_tokens,
            rate.timed_tokens,
            rate.charged_tokens,
            *(format(value, ".17g") for value in (
                rate.conditioning_seconds, rate.timed_seconds, rate.charged_seconds
            )),
        ]
    return canonical_digest(
        "optima.qualification.marginal-speed-evidence.v1",
        {
            "selected_delta_digest": selected,
            "candidate_launch_digest": candidate,
            "calibration_digest": calibration,
            "calibration_context_digest": context,
            "workload_digest": workload,
            "runtime_resource_policy_digest": runtime_policy,
            "rates": [row(rate) for rate in rates],
        },
    )


def project_marginal_speed(
    lifecycle: object,
    *,
    selected_delta_digest: str,
    calibration: object,
    expected_context: object,
    model_mount: object,
    expected_launch_resource_policy_digest: str,
    expected_runtime_resource_policy_digest: str,
    expected_device_policy_digest: str,
) -> MarginalSpeedProjection:
    from optima.eval.marginal_runtime import (
        CandidateLifecycleEvidence,
        MarginalRuntimeError,
        MarginalLifecycleEvidence,
        PreparedMarginalRuntime,
        _validate_prepared_runtime,
    )
    from optima.eval.oci_backend import TrustedArenaModelMountReceipt
    if (
        type(lifecycle) is not MarginalLifecycleEvidence
        or type(lifecycle.prepared) is not PreparedMarginalRuntime
        or type(lifecycle.candidates) is not tuple
        or not lifecycle.candidates
        or any(type(row) is not CandidateLifecycleEvidence for row in lifecycle.candidates)
    ):
        raise RawSpeedEvidenceError("marginal lifecycle evidence is malformed")
    prepared = lifecycle.prepared
    try:
        _validate_prepared_runtime(prepared)
    except (MarginalRuntimeError, OSError, TypeError, ValueError) as exc:
        raise RawSpeedEvidenceError(f"prepared runtime is invalid: {exc}") from None
    if tuple(row.candidate for row in lifecycle.candidates) != prepared.candidates:
        raise RawSpeedEvidenceError("candidate lifecycle rows were relabeled or reordered")
    digests = tuple(row.arm.selected_delta_digest for row in prepared.candidates)
    if (
        _evidence_digest(selected_delta_digest, field="selected delta")
        != selected_delta_digest
        or len(set(digests)) != len(digests)
        or digests.count(selected_delta_digest) != 1
    ):
        raise RawSpeedEvidenceError("selected candidate is absent or ambiguous")
    launch = prepared.baseline_launch
    baseline_plan = prepared.baseline_session_plan
    min_margin, k, max_noise, workload = _validate_speed_context(
        calibration, expected_context, launch, baseline_plan
    )
    if (
        type(model_mount) is not TrustedArenaModelMountReceipt
        or model_mount.digest
        != _evidence_digest(
            getattr(lifecycle.baseline_before, "arena_model_receipt_digest", None),
            field="model mount receipt",
        )
        or _evidence_digest(expected_launch_resource_policy_digest, field="launch resource policy")
        != launch.resource_policy_digest
        or _evidence_digest(expected_runtime_resource_policy_digest, field="runtime resource policy")
        != expected_runtime_resource_policy_digest
        or _evidence_digest(expected_device_policy_digest, field="device policy")
        != launch.hardware.device_policy_digest
        or (
            model_mount.arena_digest,
            model_mount.model_revision_digest,
            model_mount.model_manifest_digest,
            model_mount.model_content_digest,
        )
        != (
            launch.arena_digest,
            launch.model_revision_digest,
            launch.model_manifest_digest,
            launch.model_content_digest,
        )
    ):
        raise RawSpeedEvidenceError("validator-owned model/resource/device identity differs")
    try:
        model_mount.reopen()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise RawSpeedEvidenceError(f"model mount cannot be reopened: {exc}") from None
    state: dict[str, object] = {
        name: set()
        for name in ("sessions", "device_launches", "requests", "nonces", "runtime_policies")
    }
    def rate(
        execution: object, arm_launch: object, arm_binding: object, arm_plan: object
    ) -> ChargedExecutionRate:
        return _validate_execution_evidence(
            execution,
            launch=arm_launch,
            binding=arm_binding,
            model_mount=model_mount,
            plan=arm_plan,
            expected_resource_policy_digest=expected_runtime_resource_policy_digest,
            expected_device_policy_digest=expected_device_policy_digest,
            state=state,
        )
    before = rate(lifecycle.baseline_before, launch, prepared.incumbent_binding, baseline_plan)
    candidate_rates: dict[str, ChargedExecutionRate] = {}
    for digest, row in zip(digests, lifecycle.candidates):
        candidate_rates[digest] = rate(
            row.execution, row.candidate.launch, row.candidate.binding, row.candidate.session_plan
        )
    selected_index = digests.index(selected_delta_digest)
    selected = lifecycle.candidates[selected_index]
    candidate_rate = candidate_rates[selected_delta_digest]
    after = rate(lifecycle.baseline_after, launch, prepared.incumbent_binding, baseline_plan)
    verdict = score_speedup(
        [before.tokens_per_second, after.tokens_per_second],
        candidate_rate.tokens_per_second,
        min_margin=min_margin,
        k=k,
        max_noise=max_noise,
    )
    return MarginalSpeedProjection(
        selected_delta_digest=selected_delta_digest,
        candidate_launch_digest=selected.candidate.launch.digest,
        calibration_digest=calibration.digest,
        calibration_context_digest=expected_context.digest,
        workload_digest=workload,
        evidence_digest=_projection_digest(
            selected_delta_digest, selected.candidate.launch.digest, calibration.digest,
            expected_context.digest, workload, expected_runtime_resource_policy_digest,
            (before, candidate_rate, after)
        ),
        baseline_before=before,
        candidate=candidate_rate,
        baseline_after=after,
        verdict=verdict,
    )


def relative_spread(samples: list[float]) -> float:
    vals = [_positive_number(sample, field="baseline throughput") for sample in samples]
    if len(vals) < 2:
        return float("inf")
    mean = statistics.fmean(vals)
    if not math.isfinite(mean) or mean <= 0:
        raise RawSpeedEvidenceError("baseline mean is not finite and positive")
    if len(vals) == 2:
        return (max(vals) - min(vals)) / mean
    return statistics.pstdev(vals) / mean


def score_speedup(
    baseline_reads: list[float],
    candidate_read: float,
    *,
    min_margin: float = 0.005,
    k: float = 2.0,
    max_noise: float = 0.10,
) -> SpeedupVerdict:
    margin = _finite_time(min_margin, field="min_margin")
    multiplier = _finite_time(k, field="noise multiplier")
    noise_ceiling = _finite_time(max_noise, field="max_noise")
    if not 0 < margin < 1 or multiplier <= 0 or not 0 <= noise_ceiling < 1:
        raise RawSpeedEvidenceError("speed policy is outside its allowed range")
    reads = [
        _positive_number(sample, field="baseline throughput")
        for sample in baseline_reads
    ]
    candidate = _positive_number(candidate_read, field="candidate throughput")
    if not reads:
        return SpeedupVerdict(0.0, float("inf"), 1.0 + min_margin, False, False,
                              len(reads), "missing/zero throughput sample")
    base = statistics.fmean(reads)
    noise = relative_spread(reads)
    speedup = candidate / base
    required = 1.0 + max(margin, multiplier * (noise if math.isfinite(noise) else 0.0))
    if not math.isfinite(speedup) or not math.isfinite(required):
        raise RawSpeedEvidenceError("derived speed verdict is non-finite")
    confident = len(reads) >= 2 and noise <= noise_ceiling
    passed = confident and speedup >= required
    if not confident:
        if len(reads) < 2:
            detail = "single baseline read -> noise unmeasured; cannot crown (bookend the baseline)"
        else:
            detail = f"baseline drift {noise:.1%} > max_noise {noise_ceiling:.0%}; NO-DECISION (re-queue)"
    elif passed:
        detail = f"speedup {speedup:.3f} >= required {required:.3f} (noise {noise:.1%})"
    else:
        detail = f"speedup {speedup:.3f} < required {required:.3f} (noise {noise:.1%})"
    return SpeedupVerdict(
        speedup=speedup, noise=noise, required=required,
        passed_speedup=passed, confident=confident, n_baselines=len(reads), detail=detail,
    )
