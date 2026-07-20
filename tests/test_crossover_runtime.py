from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from optima.eval.crossover_runtime import (
    CrossoverRuntimeError,
    ResidentArmPlan,
    ResidentCrossoverPlan,
    ResidentSpeedPolicy,
    SpeedStageDecision,
    run_resident_crossover_speed,
)
from optima.eval.device_state import DeviceStatePolicy
from optima.eval.engine_launch import PhysicalHardwareBinding
from optima.eval.oci_backend import EngineExecutionEvidence, OCIEngineExecutor
from optima.eval.oci_outer_session import BatchExecutionEvidence, SessionExecutionEvidence
from optima.eval.qualification_runner import (
    QualificationStageExit,
    QualificationRunnerError,
    ResidentSpeedWitness,
    _resident_speed_projection_digest,
)
from optima.eval.qualification import QualificationDecision
from optima.settlement import ResidentLaneOrientation
from tests.test_oci_backend import _case, _manager


def _right_lane(case):
    gpu = replace(
        case.device_policy.expected_gpus[0],
        physical_id=1,
        uuid="GPU-11111111-1111-1111-1111-111111111111",
        pci_bus_id="00000000:02:00.0",
    )
    policy = replace(case.device_policy, expected_gpus=(gpu,))
    hardware = replace(
        case.launch.hardware,
        device_policy_digest=policy.policy_sha256,
    )
    physical = PhysicalHardwareBinding(
        ("1",),
        hardware.architecture,
        hardware.topology_class,
        hardware.topology_digest,
        hardware.tp_size,
        hardware.ep_size,
        hardware.dp_size,
        hardware.device_policy_digest,
    )
    launch = replace(case.launch, hardware=hardware)
    binding = replace(case.binding, physical_hardware=physical)
    plan = replace(
        case.plan,
        launch_digest=launch.digest,
        expected_preflight=replace(
            case.plan.expected_preflight,
            launch_digest=launch.digest,
        ),
    )
    return policy, launch, binding, plan


class _Controller:
    def __init__(
        self,
        plan,
        *,
        lane: str,
        timed_durations: tuple[float, ...],
        trace: list[str],
        trace_lock: threading.Lock,
        active_count: list[int],
        overlap: list[bool],
    ) -> None:
        self.plan = plan
        self.lane = lane
        self.timed_durations = timed_durations
        self.trace = trace
        self.trace_lock = trace_lock
        self.active_count = active_count
        self.overlap = overlap
        self.session_id = ("a" if lane == "left" else "b") * 32
        self.rows: list[BatchExecutionEvidence] = []
        self.clock = 10.0
        self.closed = False

    @property
    def next_batch_index(self) -> int:
        return len(self.rows)

    def execute_next(self) -> BatchExecutionEvidence:
        index = len(self.rows)
        if index >= len(self.plan.prompt_batches):
            raise CrossoverRuntimeError("fake session exhausted")
        with self.trace_lock:
            if self.active_count[0]:
                self.overlap[0] = True
            self.active_count[0] += 1
            self.trace.append(f"{self.lane}:{index}")
        try:
            time.sleep(0.001)
            role_index, local_index = divmod(index, 2)
            duration = (
                0.1 if local_index == 0 else self.timed_durations[role_index]
            )
            prompts = self.plan.prompt_batches[index]
            tokens = len(prompts) * self.plan.max_new_tokens
            started = self.clock
            self.clock += duration
            row = BatchExecutionEvidence(
                index,
                f"{index + (1 if self.lane == 'left' else 100):032x}",
                f"{index + (1000 if self.lane == 'left' else 2000):032x}",
                started,
                self.clock,
                tokens,
                SimpleNamespace(observed_tokens=tokens),
            )
            self.rows.append(row)
            return row
        finally:
            with self.trace_lock:
                self.active_count[0] -= 1

    def finish(self, *, require_all: bool = True) -> SessionExecutionEvidence:
        assert not require_all
        if len(self.rows) <= self.plan.warmup_count:
            raise CrossoverRuntimeError("fake session has no timed evidence")
        with self.trace_lock:
            self.trace.append(f"{self.lane}:close")
        self.closed = True
        first_timed = self.rows[self.plan.warmup_count]
        return SessionExecutionEvidence(
            self.session_id,
            self.plan.launch_digest,
            self.plan.expected_preflight,
            9.0,
            tuple(self.rows),
            self.plan.warmup_count,
            self.plan.conditioning_count,
            self.rows[0].request_started_at,
            first_timed.response_completed_at,
            sum(
                row.token_numerator
                for row in self.rows[: self.plan.warmup_count + 1]
            ),
            self.clock + 0.01,
        )

    def abort(self) -> None:
        self.closed = True


def _install_fake_execution(
    executor: OCIEngineExecutor,
    *,
    lane: str,
    durations: tuple[float, ...],
    trace: list[str],
    trace_lock: threading.Lock,
    active_count: list[int],
    overlap: list[bool],
) -> None:
    def execute_opened(launch, binding, mount, plan, *, deadline, driver):
        del binding, mount, deadline
        controller = _Controller(
            plan,
            lane=lane,
            timed_durations=durations,
            trace=trace,
            trace_lock=trace_lock,
            active_count=active_count,
            overlap=overlap,
        )
        session = driver(controller)
        physical_ids = executor.device_policy.physical_gpu_ids
        receipts = tuple(
            SimpleNamespace(
                completed_monotonic_s=float(index + 1),
                launch_id=("c" if lane == "left" else "d") * 32,
                selected_physical_gpu_ids=physical_ids,
                sequence=index,
                started_monotonic_s=float(index),
            )
            for index in (1, 2, 3)
        )
        return EngineExecutionEvidence(
            "optima.oci-resident-engine-execution.v1",
            launch.digest,
            SimpleNamespace(),
            "1" * 64,
            "2" * 64,
            executor.config.runtime.digest,
            SimpleNamespace(),
            ("3" if lane == "left" else "4") * 64,
            ("5" if lane == "left" else "6") * 64,
            (),
            receipts,  # type: ignore[arg-type]
            session,
        )

    executor.execute_opened = execute_opened  # type: ignore[method-assign]


def _rig(
    tmp_path: Path,
    candidate_durations: tuple[float, ...],
    *,
    distinct_runtime_policies: bool = False,
):
    left_case = _case(tmp_path / "left")
    right_case = _case(tmp_path / "right")
    right_policy, right_launch, right_binding, right_plan = _right_lane(right_case)
    left_config = left_case.config
    right_config = right_case.config
    left_launch = left_case.launch
    left_plan = left_case.plan
    if distinct_runtime_policies:
        left_runtime = replace(
            left_config.runtime,
            cpuset_cpus="0-7",
            cpuset_mems="0",
        )
        right_runtime = replace(
            right_config.runtime,
            cpuset_cpus="8-15",
            cpuset_mems="1",
        )
        left_config = replace(
            left_config,
            prebuild=replace(
                left_config.prebuild,
                policy=replace(
                    left_config.prebuild.policy,
                    runtime_policy_digest=left_runtime.digest,
                ),
            ),
            runtime=left_runtime,
        )
        right_config = replace(
            right_config,
            prebuild=replace(
                right_config.prebuild,
                policy=replace(
                    right_config.prebuild.policy,
                    runtime_policy_digest=right_runtime.digest,
                ),
            ),
            runtime=right_runtime,
        )
        left_launch = replace(
            left_launch,
            resource_policy_digest=(
                left_config.prebuild.policy.resource_policy_digest
            ),
        )
        left_plan = replace(
            left_plan,
            launch_digest=left_launch.digest,
            expected_preflight=replace(
                left_plan.expected_preflight,
                launch_digest=left_launch.digest,
            ),
        )
        right_launch = replace(
            right_launch,
            resource_policy_digest=(
                right_config.prebuild.policy.resource_policy_digest
            ),
        )
        right_plan = replace(
            right_plan,
            launch_digest=right_launch.digest,
            expected_preflight=replace(
                right_plan.expected_preflight,
                launch_digest=right_launch.digest,
            ),
        )
    baseline_executor = OCIEngineExecutor(
        left_config,
        left_case.device_policy,
        manager=_manager(left_case),
    )
    candidate_executor = OCIEngineExecutor(
        right_config,
        right_policy,
        manager=_manager(right_case),
    )
    baseline = ResidentArmPlan(
        left_launch,
        left_case.binding,
        left_plan,
        baseline_executor.manager.namespace_digest,
        baseline_executor.config.runtime.digest,
        baseline_executor.device_policy.configuration_sha256,
    )
    candidate = ResidentArmPlan(
        right_launch,
        right_binding,
        right_plan,
        candidate_executor.manager.namespace_digest,
        candidate_executor.config.runtime.digest,
        candidate_executor.device_policy.configuration_sha256,
    )
    trace: list[str] = []
    trace_lock = threading.Lock()
    active_count = [0]
    overlap = [False]
    _install_fake_execution(
        baseline_executor,
        lane="left",
        durations=(1.0, 1.0, 1.0),
        trace=trace,
        trace_lock=trace_lock,
        active_count=active_count,
        overlap=overlap,
    )
    _install_fake_execution(
        candidate_executor,
        lane="right",
        durations=candidate_durations,
        trace=trace,
        trace_lock=trace_lock,
        active_count=active_count,
        overlap=overlap,
    )
    plan = ResidentCrossoverPlan(
        "7" * 64,
        baseline,
        candidate,
        ResidentSpeedPolicy(60, 0.005, 2.0, 0.1, "8" * 64, "9" * 64),
    )
    return (
        plan,
        baseline_executor,
        candidate_executor,
        left_case.mount,
        trace,
        overlap,
    )


@pytest.mark.parametrize(
    ("candidate_durations", "expected_decision"),
    (
        ((1.02, 1.02), SpeedStageDecision.FAIL),
        ((0.90, 0.90), SpeedStageDecision.PASS),
    ),
)
def test_clear_result_stops_after_three_serialized_reads(
    tmp_path: Path,
    candidate_durations: tuple[float, ...],
    expected_decision: SpeedStageDecision,
) -> None:
    plan, baseline, candidate, mount, trace, overlap = _rig(
        tmp_path, candidate_durations
    )
    result = run_resident_crossover_speed(
        plan,
        baseline_executor=baseline,
        candidate_executor=candidate,
        model_mount=mount,
        deadline=time.monotonic() + 60,
    )

    assert result.decision is expected_decision
    assert not result.escalated
    assert tuple(row.role for row in result.rates) == ("B", "C", "B_prime")
    assert trace[:6] == [
        "left:0",
        "left:1",
        "right:0",
        "right:1",
        "left:2",
        "left:3",
    ]
    assert trace.index("right:close") > trace.index("left:3")
    assert not overlap[0]
    assert len(result.baseline_execution.session.batches) == 4
    assert len(result.candidate_execution.session.batches) == 2
    assert result.regrade(plan) == result.final_verdict
    assert result.digest


def test_borderline_result_adds_only_candidate_and_baseline_repeat(
    tmp_path: Path,
) -> None:
    plan, baseline, candidate, mount, trace, overlap = _rig(
        tmp_path, (0.993, 0.993)
    )
    result = run_resident_crossover_speed(
        plan,
        baseline_executor=baseline,
        candidate_executor=candidate,
        model_mount=mount,
        deadline=time.monotonic() + 60,
    )

    assert result.escalated
    assert result.decision is SpeedStageDecision.PASS
    assert tuple(row.role for row in result.rates) == (
        "B",
        "C",
        "B_prime",
        "C_prime",
        "B_double_prime",
    )
    assert trace.index("right:2") < trace.index("left:4")
    assert trace.index("right:close") > trace.index("left:5")
    assert not overlap[0]
    assert len(result.baseline_execution.session.batches) == 6
    assert len(result.candidate_execution.session.batches) == 4


def test_plan_rejects_overlapping_physical_lanes(tmp_path: Path) -> None:
    case = _case(tmp_path)
    executor = OCIEngineExecutor(
        case.config,
        case.device_policy,
        manager=_manager(case),
    )
    arm = ResidentArmPlan(
        case.launch,
        case.binding,
        case.plan,
        executor.manager.namespace_digest,
        executor.config.runtime.digest,
        executor.device_policy.configuration_sha256,
    )
    with pytest.raises(CrossoverRuntimeError, match="overlap"):
        ResidentCrossoverPlan(
            "7" * 64,
            arm,
            arm,
            ResidentSpeedPolicy(60, 0.005, 2.0, 0.1, "8" * 64, "9" * 64),
        )


def test_retained_rate_span_is_independently_regraded(tmp_path: Path) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path, (0.90, 0.90)
    )
    result = run_resident_crossover_speed(
        plan,
        baseline_executor=baseline,
        candidate_executor=candidate,
        model_mount=mount,
        deadline=time.monotonic() + 60,
    )
    first = result.rates[0]
    changed_seconds = first.timed_seconds * 2
    changed_charged_seconds = first.conditioning_seconds + changed_seconds
    tampered = replace(
        result,
        rates=(
            replace(
                first,
                timed_seconds=changed_seconds,
                charged_seconds=changed_charged_seconds,
                tokens_per_second=first.charged_tokens / changed_charged_seconds,
            ),
            *result.rates[1:],
        ),
    )

    with pytest.raises(CrossoverRuntimeError, match="independently regrade"):
        tampered.regrade(plan)


@pytest.mark.parametrize("candidate_durations", ((0.90, 0.90), (1.02, 1.02)))
def test_resident_speed_witness_round_trips_pass_or_fail_raw_stage(
    tmp_path: Path, candidate_durations: tuple[float, ...]
) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path, candidate_durations
    )
    result = run_resident_crossover_speed(
        plan,
        baseline_executor=baseline,
        candidate_executor=candidate,
        model_mount=mount,
        deadline=time.monotonic() + 60,
    )

    witness = ResidentSpeedWitness.from_evidence(result, plan)
    assert ResidentSpeedWitness.from_dict(witness.to_dict()) == witness
    assert witness.policy.version == 3
    assert witness.started_monotonic_s == result.started_monotonic_s
    assert witness.completed_monotonic_s == result.completed_monotonic_s
    assert witness.resident_policy.max_qualification_seconds == 7_200

    if result.decision is not SpeedStageDecision.PASS:
        decision = (
            QualificationDecision.NO_DECISION
            if result.decision is SpeedStageDecision.NO_DECISION
            else QualificationDecision.FAIL
        )
        stage = QualificationStageExit(
            "a" * 64,
            "b" * 64,
            plan.selected_delta_digest,
            "speed",
            decision,
            "speed_noise"
            if decision is QualificationDecision.NO_DECISION
            else "speed_regression",
            witness,
            None,
            None,
            None,
            None,
        )
        assert QualificationStageExit.from_dict(stage.to_dict()) == stage

    tampered = witness.to_dict()
    tampered["rates"][0]["lane_digest"] = plan.candidate_lane_digest
    with pytest.raises(QualificationRunnerError, match="digest"):
        ResidentSpeedWitness.from_dict(tampered)


def test_resident_speed_witness_binds_distinct_numa_lane_policies(
    tmp_path: Path,
) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path,
        (0.90, 0.90),
        distinct_runtime_policies=True,
    )
    assert baseline.config.runtime.cpuset_cpus == "0-7"
    assert baseline.config.runtime.cpuset_mems == "0"
    assert candidate.config.runtime.cpuset_cpus == "8-15"
    assert candidate.config.runtime.cpuset_mems == "1"
    assert (
        plan.baseline.runtime_resource_policy_digest
        != plan.candidate.runtime_resource_policy_digest
    )
    assert (
        plan.baseline.launch.resource_policy_digest
        != plan.candidate.launch.resource_policy_digest
    )

    result = run_resident_crossover_speed(
        plan,
        baseline_executor=baseline,
        candidate_executor=candidate,
        model_mount=mount,
        deadline=time.monotonic() + 60,
    )
    witness = ResidentSpeedWitness.from_evidence(result, plan)

    assert witness.baseline_runtime_resource_policy_digest == (
        plan.baseline.runtime_resource_policy_digest
    )
    assert witness.candidate_runtime_resource_policy_digest == (
        plan.candidate.runtime_resource_policy_digest
    )
    assert ResidentSpeedWitness.from_dict(witness.to_dict()) == witness

    tampered = witness.to_dict()
    tampered["baseline_runtime_resource_policy_digest"] = (
        witness.candidate_runtime_resource_policy_digest
    )
    with pytest.raises(QualificationRunnerError, match="digest"):
        ResidentSpeedWitness.from_dict(tampered)


def test_resident_settlement_control_accepts_exact_lane_policy_swap(
    tmp_path: Path,
) -> None:
    plan, baseline, candidate, mount, _trace, _overlap = _rig(
        tmp_path,
        (0.90, 0.90),
        distinct_runtime_policies=True,
    )
    result = run_resident_crossover_speed(
        plan,
        baseline_executor=baseline,
        candidate_executor=candidate,
        model_mount=mount,
        deadline=time.monotonic() + 60,
    )
    primary_witness = ResidentSpeedWitness.from_evidence(result, plan)
    reproduction_policy = replace(
        primary_witness.resident_policy,
        calibration_digest="2" * 64,
        calibration_context_digest="3" * 64,
    )
    swapped_rates = tuple(
        replace(
            row,
            lane_digest=(
                primary_witness.candidate_lane_digest
                if row.role.startswith("B")
                else primary_witness.baseline_lane_digest
            ),
        )
        for row in primary_witness.rates
    )
    swapped_fields = {
        "selected_delta_digest": primary_witness.selected_delta_digest,
        "candidate_launch_digest": primary_witness.candidate_launch_digest,
        "calibration_digest": reproduction_policy.calibration_digest,
        "calibration_context_digest": (
            reproduction_policy.calibration_context_digest
        ),
        "workload_digest": primary_witness.workload_digest,
        "baseline_runtime_resource_policy_digest": (
            primary_witness.candidate_runtime_resource_policy_digest
        ),
        "candidate_runtime_resource_policy_digest": (
            primary_witness.baseline_runtime_resource_policy_digest
        ),
        "plan_digest": "d" * 64,
        "baseline_lane_digest": primary_witness.candidate_lane_digest,
        "candidate_lane_digest": primary_witness.baseline_lane_digest,
        "baseline_quiescence_digest": "e" * 64,
        "candidate_quiescence_digest": "f" * 64,
        "raw_crossover_digest": "1" * 64,
        "resident_policy": reproduction_policy,
        "rates": swapped_rates,
        "started_monotonic_s": primary_witness.started_monotonic_s,
        "completed_monotonic_s": primary_witness.completed_monotonic_s,
    }
    reproduction_witness = ResidentSpeedWitness(
        **swapped_fields,
        evidence_digest=_resident_speed_projection_digest(**swapped_fields),
    )

    primary = ResidentLaneOrientation.from_resident_speed_witness(
        primary_witness
    )
    reproduction = ResidentLaneOrientation.from_resident_speed_witness(
        reproduction_witness
    )
    assert reproduction.control_digest == primary.control_digest
    assert reproduction.is_exact_swap_of(primary)


def test_resident_policy_binds_total_qualification_budget() -> None:
    policy = ResidentSpeedPolicy(
        600,
        0.005,
        2.0,
        0.1,
        "8" * 64,
        "9" * 64,
        max_qualification_seconds=1_800,
    )
    assert ResidentSpeedPolicy.from_dict(policy.to_dict()) == policy
    with pytest.raises(CrossoverRuntimeError, match="unsupported"):
        replace(policy, max_qualification_seconds=599)
