from __future__ import annotations

import dataclasses

import pytest

from optima.eval.device_state import (
    CommandResult,
    DeviceStateCancelledError,
    DeviceStateClockError,
    DeviceStateConfigurationError,
    DeviceStateEnvelopeError,
    DeviceStateGuard,
    DeviceStateParseError,
    DeviceStatePolicy,
    DeviceStatePolicyError,
    DeviceStateTimeoutError,
    GPUConfiguration,
    NVIDIA_SMI,
    provision_gpu_configurations,
    validate_device_state_policy,
)
from optima.eval.engine_launch import LogicalHardwareSpec, PhysicalHardwareBinding


def _gpu(physical_id: int) -> GPUConfiguration:
    suffix = f"{physical_id:012d}"
    return GPUConfiguration(
        physical_id=physical_id,
        uuid=f"GPU-00000000-0000-0000-0000-{suffix}",
        pci_bus_id=f"00000000:{physical_id + 1:02x}:00.0",
        name="NVIDIA B300 SXM6 AC",
        memory_total_mib=275_040,
        driver_version="595.71.05",
        power_limit_mw=1_000_000,
        compute_mode="Default",
        persistence_mode="Enabled",
        application_graphics_clock_mhz=None,
        application_memory_clock_mhz=None,
        max_graphics_clock_mhz=2_100,
        max_memory_clock_mhz=4_000,
    )


def _policy(**changes) -> DeviceStatePolicy:
    base = DeviceStatePolicy(
        expected_gpus=(_gpu(0), _gpu(1)),
        maximum_temperature_c=60,
        maximum_gpu_utilization_percent=5,
        maximum_memory_utilization_percent=5,
        allowed_active_pstates=("P0",),
        active_maximum_graphics_clock_mhz=300,
        active_memory_clock_mhz=405,
        active_maximum_power_draw_mw=300_000,
        required_consecutive_idle_samples=2,
        poll_interval_s=1.0,
        ready_poll_interval_s=0.1,
        drain_timeout_s=10.0,
        maximum_samples=16,
    )
    return dataclasses.replace(base, **changes)


def _hardware(policy: DeviceStatePolicy, *, selected: tuple[str, ...] = ("0", "1")):
    logical = LogicalHardwareSpec(
        visible_gpu_count=2,
        architecture="sm120",
        topology_class="pcie",
        topology_digest="a" * 64,
        tp_size=2,
        ep_size=1,
        dp_size=1,
        device_policy_digest=policy.policy_sha256,
    )
    physical = PhysicalHardwareBinding(
        physical_gpu_ids=selected,
        architecture=logical.architecture,
        topology_class=logical.topology_class,
        topology_digest=logical.topology_digest,
        tp_size=logical.tp_size,
        ep_size=logical.ep_size,
        dp_size=logical.dp_size,
        device_policy_digest=logical.device_policy_digest,
    )
    return logical, physical


def test_device_policy_binds_exact_indices_or_uuids_to_logical_hardware():
    policy = _policy()
    logical, physical = _hardware(policy)
    validate_device_state_policy(
        policy, logical_hardware=logical, physical_hardware=physical
    )

    uuids = tuple(gpu.uuid for gpu in policy.expected_gpus)
    logical, physical = _hardware(policy, selected=uuids)
    validate_device_state_policy(
        policy, logical_hardware=logical, physical_hardware=physical
    )

    logical, physical = _hardware(policy, selected=("2", "3"))
    with pytest.raises(DeviceStatePolicyError, match="identities"):
        validate_device_state_policy(
            policy, logical_hardware=logical, physical_hardware=physical
        )


def _gpu_row(
    gpu: GPUConfiguration,
    *,
    physical_id: int | None = None,
    power_limit_mw: int | None = None,
    temperature: int = 35,
    gpu_utilization: int = 0,
    memory_utilization: int = 0,
    persistence_mode: str | None = None,
    max_graphics_clock_mhz: int | None = None,
    pstate: str = "P0",
    current_graphics_clock_mhz: int = 210,
    current_memory_clock_mhz: int = 405,
    power_draw_mw: int = 80_500,
) -> str:
    return ", ".join(
        (
            str(gpu.physical_id if physical_id is None else physical_id),
            gpu.uuid,
            gpu.pci_bus_id,
            gpu.name,
            str(gpu.memory_total_mib),
            gpu.driver_version,
            f"{(gpu.power_limit_mw if power_limit_mw is None else power_limit_mw) / 1000:.3f}",
            gpu.compute_mode,
            persistence_mode or gpu.persistence_mode,
            (
                "[Requested functionality has been deprecated]"
                if gpu.application_graphics_clock_mhz is None
                else str(gpu.application_graphics_clock_mhz)
            ),
            (
                "[Requested functionality has been deprecated]"
                if gpu.application_memory_clock_mhz is None
                else str(gpu.application_memory_clock_mhz)
            ),
            str(
                gpu.max_graphics_clock_mhz
                if max_graphics_clock_mhz is None
                else max_graphics_clock_mhz
            ),
            str(gpu.max_memory_clock_mhz),
            pstate,
            str(temperature),
            str(gpu_utilization),
            str(memory_utilization),
            str(current_graphics_clock_mhz),
            str(current_memory_clock_mhz),
            f"{power_draw_mw / 1000:.3f}",
        )
    )


def _gpu_output(policy: DeviceStatePolicy, **row_changes) -> str:
    return "\n".join(
        _gpu_row(gpu, **row_changes) for gpu in policy.expected_gpus
    ) + "\n"


def _idle_process_output(policy: DeviceStatePolicy) -> str:
    rows = ["# gpu pid type sm mem enc dec command"]
    rows.extend(f"{physical_id} - - - - - - -" for physical_id in policy.physical_gpu_ids)
    return "\n".join(rows) + "\n"


class FakeClock:
    def __init__(self, value: float = 100.0):
        self.value = value

    def __call__(self) -> float:
        return self.value

    def sleep(self, duration: float) -> None:
        self.value += duration


class ScriptedRunner:
    def __init__(
        self,
        gpu_outputs: list[str],
        process_outputs: list[str],
        *,
        clock: FakeClock | None = None,
        command_duration_s: float = 0.0,
    ):
        self.gpu_outputs = gpu_outputs
        self.process_outputs = process_outputs
        self.clock = clock
        self.command_duration_s = command_duration_s
        self.gpu_index = 0
        self.process_index = 0
        self.calls: list[tuple[tuple[str, ...], float, float | None]] = []

    @staticmethod
    def _next(values: list[str], index: int) -> str:
        if not values:
            raise AssertionError("script has no output")
        return values[min(index, len(values) - 1)]

    def __call__(self, argv: tuple[str, ...], *, timeout_s: float) -> CommandResult:
        started = self.clock.value if self.clock is not None else None
        self.calls.append((argv, timeout_s, started))
        if self.clock is not None:
            self.clock.value += self.command_duration_s
        if len(argv) > 1 and argv[1] == "pmon":
            output = self._next(self.process_outputs, self.process_index)
            self.process_index += 1
        else:
            output = self._next(self.gpu_outputs, self.gpu_index)
            self.gpu_index += 1
        return CommandResult(0, output)


def _guard(
    policy: DeviceStatePolicy,
    *,
    runner: ScriptedRunner,
    clock: FakeClock,
) -> DeviceStateGuard:
    return DeviceStateGuard(
        policy,
        runner=runner,
        clock=clock,
        sleep=clock.sleep,
    )


def test_successful_pre_and_post_receipts_are_monotonic_and_query_exact_ids():
    policy = _policy()
    clock = FakeClock()
    runner = ScriptedRunner(
        [_gpu_output(policy)], [_idle_process_output(policy)], clock=clock
    )
    guard = _guard(policy, runner=runner, clock=clock)

    before = guard.before_launch("launch-c", deadline=clock.value + 20)
    after = guard.after_launch("launch-c", deadline=clock.value + 20)

    assert (before.sequence, after.sequence) == (1, 2)
    assert before.phase == "pre" and after.phase == "post"
    assert before.selected_physical_gpu_ids == (0, 1)
    assert before.configuration_sha256 == policy.configuration_sha256
    assert before.completed_monotonic_s <= after.started_monotonic_s
    assert all(sample.idle for sample in (*before.samples, *after.samples))
    assert all(len(receipt.samples) == 2 for receipt in (before, after))

    for argv, timeout_s, _ in runner.calls:
        assert argv[0] == NVIDIA_SMI
        assert "--id=0,1" in argv
        assert timeout_s > 0
        assert not any(
            argument.startswith(("--lock", "-lgc", "-lmc", "-pl", "--reset"))
            for argument in argv
        )


@pytest.mark.parametrize(
    "gpu_output",
    (
        "malformed\n",
        "0,too,few,fields\n",
    ),
)
def test_malformed_gpu_query_fails_closed(gpu_output):
    policy = _policy()
    clock = FakeClock()
    runner = ScriptedRunner([gpu_output], [_idle_process_output(policy)])
    with pytest.raises(DeviceStateParseError):
        _guard(policy, runner=runner, clock=clock).before_launch("launch-b")


@pytest.mark.parametrize("shape", ("missing", "extra", "duplicate"))
def test_missing_extra_and_duplicate_physical_gpus_fail_closed(shape):
    policy = _policy()
    rows = [_gpu_row(gpu) for gpu in policy.expected_gpus]
    if shape == "missing":
        rows = rows[:1]
    elif shape == "extra":
        rows.append(_gpu_row(_gpu(2)))
    else:
        rows[1] = rows[0]
    clock = FakeClock()
    runner = ScriptedRunner(
        ["\n".join(rows) + "\n"], [_idle_process_output(policy)]
    )
    with pytest.raises(DeviceStateConfigurationError):
        _guard(policy, runner=runner, clock=clock).before_launch("launch-b")


def test_compute_and_graphics_processes_reset_idle_consecutive_count():
    policy = _policy(required_consecutive_idle_samples=2)
    clock = FakeClock()
    active = "\n".join(
        (
            "# gpu pid type sm mem enc dec command",
            "0 111 C 2 0 - - python",
            "0 222 G 0 1 - - compositor",
            "1 - - - - - - -",
        )
    ) + "\n"
    idle = _idle_process_output(policy)
    runner = ScriptedRunner(
        [_gpu_output(policy)], [active, idle, idle], clock=clock
    )

    receipt = _guard(policy, runner=runner, clock=clock).before_launch("launch-bookend")

    assert len(receipt.samples) == 3
    assert not receipt.samples[0].idle
    assert {(item.pid, item.kind) for item in receipt.samples[0].processes} == {
        (111, "C"),
        (222, "G"),
    }
    assert [sample.idle for sample in receipt.samples] == [False, True, True]


def test_configuration_mutation_between_idle_samples_is_terminal():
    policy = _policy(required_consecutive_idle_samples=2)
    clock = FakeClock()
    changed = "\n".join(
        _gpu_row(gpu, power_limit_mw=900_000) for gpu in policy.expected_gpus
    ) + "\n"
    runner = ScriptedRunner(
        [_gpu_output(policy), changed],
        [_idle_process_output(policy)],
        clock=clock,
    )

    with pytest.raises(DeviceStateConfigurationError, match="power_limit_mw"):
        _guard(policy, runner=runner, clock=clock).before_launch("launch-c")


def test_active_processes_time_out_under_one_absolute_drain_deadline():
    policy = _policy(drain_timeout_s=2.5, poll_interval_s=1.0, maximum_samples=16)
    clock = FakeClock()
    active = "\n".join(
        (
            "# gpu pid type sm mem enc dec command",
            "0 333 C 0 0 - - python",
            "1 - - - - - - -",
        )
    ) + "\n"
    runner = ScriptedRunner([_gpu_output(policy)], [active], clock=clock)

    with pytest.raises(DeviceStateTimeoutError, match="deadline|idle envelope"):
        _guard(policy, runner=runner, clock=clock).before_launch("launch-b")
    assert clock.value <= 102.5


def test_absolute_deadline_is_forwarded_as_decreasing_subprocess_timeouts():
    policy = _policy(required_consecutive_idle_samples=2, poll_interval_s=0.5)
    clock = FakeClock(50.0)
    runner = ScriptedRunner(
        [_gpu_output(policy)],
        [_idle_process_output(policy)],
        clock=clock,
        command_duration_s=0.2,
    )
    deadline = 55.0

    _guard(policy, runner=runner, clock=clock).before_launch(
        "launch-b", deadline=deadline
    )

    timeouts = [timeout for _, timeout, _ in runner.calls]
    assert timeouts == sorted(timeouts, reverse=True)
    for (_, timeout, started) in runner.calls:
        assert started is not None
        assert timeout <= deadline - started + 1e-9


def test_active_conditioning_requires_consecutive_pinned_envelope_samples():
    policy = _policy(required_consecutive_idle_samples=2)
    clock = FakeClock()
    active = "\n".join(
        (
            "# gpu pid type sm mem enc dec command",
            "0 501 C 90 40 - - rank0",
            "1 502 C 90 40 - - rank1",
        )
    ) + "\n"
    unconditioned = _gpu_output(
        policy,
        pstate="P8",
        current_graphics_clock_mhz=500,
        gpu_utilization=99,
        memory_utilization=99,
    )
    conditioned = _gpu_output(
        policy,
        pstate="P0",
        current_graphics_clock_mhz=210,
        gpu_utilization=99,
        memory_utilization=99,
    )
    runner = ScriptedRunner(
        [unconditioned, conditioned, conditioned],
        [active],
        clock=clock,
    )

    receipt = _guard(policy, runner=runner, clock=clock).condition_active(
        "launch-c", deadline=clock.value + 10
    )

    assert len(receipt.samples) == 3
    assert not receipt.samples[0].active_envelope_passed
    assert "pstate=P8" in receipt.samples[0].active_envelope_reason
    assert [sample.active_envelope_passed for sample in receipt.samples] == [
        False,
        True,
        True,
    ]
    # Utilization is recorded but is not forced to the idle <=5% threshold.
    assert receipt.samples[-1].telemetry[0].gpu_utilization_percent == 99


def test_active_conditioning_pre_query_cancel_consumes_no_command_or_sequence():
    policy = _policy(required_consecutive_idle_samples=2)
    clock = FakeClock()
    active = "\n".join(
        (
            "# gpu pid type sm mem enc dec command",
            "0 501 C 90 40 - - rank0",
            "1 502 C 90 40 - - rank1",
        )
    ) + "\n"
    runner = ScriptedRunner(
        [_gpu_output(policy, pstate="P0")],
        [active],
        clock=clock,
    )
    guard = _guard(policy, runner=runner, clock=clock)

    with pytest.raises(DeviceStateCancelledError, match="cancelled"):
        guard.condition_active(
            "launch-c",
            deadline=clock.value + 10,
            cancel=lambda: True,
        )
    assert runner.calls == []

    receipt = guard.condition_active(
        "launch-c",
        deadline=clock.value + 10,
    )
    assert receipt.sequence == 1


def test_active_conditioning_carries_active_run_but_requires_post_release_ready_sample():
    policy = _policy(required_consecutive_idle_samples=2)
    clock = FakeClock()
    active = "\n".join(
        (
            "# gpu pid type sm mem enc dec command",
            "0 501 C 90 40 - - rank0",
            "1 502 C 90 40 - - rank1",
        )
    ) + "\n"
    active_conditioned = _gpu_output(
        policy,
        pstate="P0",
        current_graphics_clock_mhz=210,
        gpu_utilization=99,
        memory_utilization=99,
    )
    ready_conditioned = _gpu_output(
        policy,
        pstate="P0",
        current_graphics_clock_mhz=210,
        gpu_utilization=0,
        memory_utilization=0,
    )
    runner = ScriptedRunner(
        [
            active_conditioned,
            active_conditioned,
            ready_conditioned,
            ready_conditioned,
        ],
        [active],
        clock=clock,
    )
    release_checks = 0

    def released():
        nonlocal release_checks
        release_checks += 1
        return release_checks >= 5

    receipt = _guard(policy, runner=runner, clock=clock).condition_active(
        "launch-c",
        "final-warmup-conditioning",
        deadline=clock.value + 20,
        release=released,
    )

    # Two valid concurrent samples establish the active run, but authority is
    # withheld until one stricter post-response ready sample also passes.
    assert len(receipt.samples) == 3
    assert receipt.consecutive_active_samples == 2
    assert receipt.release_sample_index == 2
    assert receipt.post_release_ready_samples == 1
    assert release_checks == 5
    assert all(sample.active_envelope_passed for sample in receipt.samples)
    assert receipt.samples[-1].telemetry[0].gpu_utilization_percent == 0


def test_active_conditioning_release_interrupts_poll_sleep_but_still_queries_ready():
    policy = _policy(required_consecutive_idle_samples=2, poll_interval_s=5.0)
    clock = FakeClock()
    active = "\n".join(
        (
            "# gpu pid type sm mem enc dec command",
            "0 501 C 90 40 - - rank0",
            "1 502 C 90 40 - - rank1",
        )
    ) + "\n"
    conditioned = _gpu_output(
        policy,
        pstate="P0",
        current_graphics_clock_mhz=210,
        gpu_utilization=0,
        memory_utilization=0,
    )
    runner = ScriptedRunner([conditioned], [active], clock=clock)
    released = False
    waits = 0

    def is_released():
        return released

    def wait_for_release(timeout_s: float):
        nonlocal released, waits
        assert timeout_s == 5.0
        waits += 1
        if waits == 2:
            released = True
        return released

    receipt = _guard(policy, runner=runner, clock=clock).condition_active(
        "launch-c",
        "final-warmup-conditioning",
        deadline=clock.value + 20,
        release=is_released,
        wait_for_release=wait_for_release,
    )

    assert len(receipt.samples) == 3
    assert receipt.release_sample_index == 2
    assert len(runner.calls) == 6  # two active queries, then one ready query
    assert clock.value == 100.0  # the five-second poll sleep was interrupted


def test_failed_post_release_ready_sample_preserves_pre_release_authority():
    policy = _policy(required_consecutive_idle_samples=2)
    clock = FakeClock()
    active = "\n".join(
        (
            "# gpu pid type sm mem enc dec command",
            "0 501 C 90 40 - - rank0",
            "1 502 C 90 40 - - rank1",
        )
    ) + "\n"
    active_conditioned = _gpu_output(
        policy,
        pstate="P0",
        current_graphics_clock_mhz=210,
        gpu_utilization=99,
        memory_utilization=99,
    )
    ready_conditioned = _gpu_output(
        policy,
        pstate="P0",
        current_graphics_clock_mhz=210,
        gpu_utilization=0,
        memory_utilization=0,
    )
    runner = ScriptedRunner(
        [
            active_conditioned,
            active_conditioned,
            active_conditioned,  # first post-release sample is not quiescent
            ready_conditioned,
            ready_conditioned,
        ],
        [active],
        clock=clock,
    )
    release_checks = 0

    def released():
        nonlocal release_checks
        release_checks += 1
        return release_checks >= 5

    receipt = _guard(policy, runner=runner, clock=clock).condition_active(
        "launch-c",
        "final-warmup-conditioning",
        deadline=clock.value + 20,
        release=released,
    )

    assert len(receipt.samples) == 4
    assert not receipt.samples[2].active_envelope_passed
    assert receipt.consecutive_active_samples == 2
    assert receipt.release_sample_index == 2
    assert receipt.post_release_ready_samples == 1
    assert receipt.samples[-1].active_envelope_passed


def test_release_before_any_concurrent_sample_cannot_mint_active_receipt():
    policy = _policy(required_consecutive_idle_samples=2)
    clock = FakeClock()
    runner = ScriptedRunner(
        [_gpu_output(policy)], [_idle_process_output(policy)], clock=clock
    )

    with pytest.raises(DeviceStateEnvelopeError, match="pre-release active"):
        _guard(policy, runner=runner, clock=clock).condition_active(
            "launch-c",
            "final-warmup-conditioning",
            deadline=clock.value + 20,
            release=lambda: True,
        )

    assert runner.calls == []


def test_long_warmup_caps_claimed_active_run_at_policy_requirement():
    policy = _policy(
        required_consecutive_idle_samples=2,
        maximum_samples=64,
        drain_timeout_s=60.0,
    )
    clock = FakeClock()
    active = "\n".join(
        (
            "# gpu pid type sm mem enc dec command",
            "0 501 C 90 40 - - rank0",
            "1 502 C 90 40 - - rank1",
        )
    ) + "\n"
    conditioned = _gpu_output(
        policy,
        pstate="P0",
        current_graphics_clock_mhz=210,
        gpu_utilization=0,
        memory_utilization=0,
    )
    runner = ScriptedRunner([conditioned], [active], clock=clock)
    release_checks = 0

    def released():
        nonlocal release_checks
        release_checks += 1
        # Two checks per pre-release query, then one at the next loop head.
        return release_checks >= 71

    receipt = _guard(policy, runner=runner, clock=clock).condition_active(
        "launch-c",
        "final-warmup-conditioning",
        deadline=clock.value + 60,
        release=released,
    )

    assert receipt.release_sample_index == 35
    assert receipt.consecutive_active_samples == 2
    assert receipt.post_release_ready_samples == 1


def test_provisioning_query_returns_exact_selected_physical_configurations():
    policy = _policy()
    clock = FakeClock(10.0)
    runner = ScriptedRunner([_gpu_output(policy)], [], clock=clock)

    configurations = provision_gpu_configurations(
        (0, 1), deadline=15.0, runner=runner, clock=clock
    )

    assert configurations == policy.expected_gpus
    assert len(runner.calls) == 1
    argv, timeout_s, started = runner.calls[0]
    assert argv[0] == NVIDIA_SMI and "--id=0,1" in argv
    assert timeout_s == 5.0 and started == 10.0


def test_malformed_process_monitor_and_clock_regression_fail_closed():
    policy = _policy()
    clock = FakeClock()
    malformed = ScriptedRunner([_gpu_output(policy)], ["unexpected output\n"])
    with pytest.raises(DeviceStateParseError):
        _guard(policy, runner=malformed, clock=clock).before_launch("launch-b")

    healthy = ScriptedRunner(
        [_gpu_output(policy)], [_idle_process_output(policy)], clock=clock
    )
    guard = _guard(policy, runner=healthy, clock=clock)
    guard.before_launch("launch-b")
    clock.value -= 100
    with pytest.raises(DeviceStateClockError):
        guard.after_launch("launch-b")
