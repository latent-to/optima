from __future__ import annotations

from dataclasses import dataclass

import pytest

from optima.artifact_abi import (
    ArtifactAggregateComponent,
    ArtifactBinding,
    ArtifactPrelaunch,
    ArtifactResource,
    ArtifactResourcePlan,
    ArtifactShapeExtent,
    ArtifactShapeFactor,
    COLLECTIVE_ALL_REDUCE_CALL_ABI,
    MOE_FUSED_EXPERTS_CALL_ABI,
    MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
    SILU_AND_MUL_CALL_ABI,
    SlotCallABI,
    SlotResource,
)
from optima.artifact_runtime import (
    ArtifactAllocationBudget,
    ArtifactPreparedState,
    ArtifactRuntimeEntry,
    ArtifactRuntimeError,
    ArtifactRuntimeLimits,
    ArtifactRuntimeProvider,
    ArtifactRuntimeStep,
    artifact_call_resources,
)


class FakeTensor:
    def __init__(
        self,
        name: str,
        shape: tuple[int, ...],
        *,
        stride: tuple[int, ...] | None = None,
        pointer: int = 16,
        storage_offset: int = 0,
    ) -> None:
        self.name = name
        self.shape = shape
        self._stride = stride or self._contiguous_stride(shape)
        self._pointer = pointer
        self._storage_offset = storage_offset
        self.fills: list[object] = []

    @staticmethod
    def _contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
        result: list[int] = []
        running = 1
        for dimension in reversed(shape):
            result.append(running)
            running *= dimension
        return tuple(reversed(result))

    def stride(self) -> tuple[int, ...]:
        return self._stride

    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result

    def storage_offset(self) -> int:
        return self._storage_offset

    def data_ptr(self) -> int:
        return self._pointer

    def fill_(self, value: object) -> "FakeTensor":
        self.fills.append(value)
        return self


@dataclass(frozen=True)
class FakeGroup:
    rank_value: int = 0
    size_value: int = 4

    def rank(self) -> int:
        return self.rank_value

    def size(self) -> int:
        return self.size_value


def _provider(
    *,
    group_pointers=None,
    capture=lambda: False,
    scope=lambda: ("fake-cuda", 0, 7),
    synchronize=lambda: None,
) -> ArtifactRuntimeProvider:
    pointers = dict(group_pointers or {})
    next_pointer = 1 << 20

    def aligned(value: object, binding: ArtifactBinding) -> None:
        if binding.assumed_align is None:
            return
        pointer = value.data_ptr()  # type: ignore[attr-defined]
        if pointer % binding.assumed_align:
            raise ArtifactRuntimeError(
                f"tensor pointer is not {binding.assumed_align}-byte aligned"
            )

    def descriptor(value: object, binding: ArtifactBinding) -> object:
        aligned(value, binding)
        return ("descriptor", value, binding.unsqueeze, binding.leading_dim)

    def tensor_pointer(value: object, binding: ArtifactBinding) -> object:
        aligned(value, binding)
        return ("device_ptr", value.data_ptr())  # type: ignore[attr-defined]

    def group_pointer(
        group: object,
        projection: str,
        binding: ArtifactBinding,
        peer_resource: object | None,
    ) -> object:
        try:
            resolver = pointers[projection]
        except KeyError:
            raise ArtifactRuntimeError(
                f"provider cannot materialize group projection {projection!r}"
            ) from None
        return resolver(group, binding, peer_resource)

    def allocate_tensor(shape, dtype, alignment):
        nonlocal next_pointer
        pointer = (next_pointer + alignment - 1) // alignment * alignment
        next_pointer = pointer + (1 << 20)
        return FakeTensor(f"{dtype}:{shape}", tuple(shape), pointer=pointer)

    def validate_allocation(value, shape, _dtype, alignment):
        if tuple(value.shape) != tuple(shape) or value.data_ptr() % alignment:
            raise ArtifactRuntimeError("fake allocation validation failed")

    return ArtifactRuntimeProvider(
        provider="fake.object.v1",
        tensor_descriptor=descriptor,
        tensor_pointer=tensor_pointer,
        current_stream=lambda: ("stream", 7),
        group_rank=lambda group: group.rank(),  # type: ignore[attr-defined]
        group_size=lambda group: group.size(),  # type: ignore[attr-defined]
        group_pointer=group_pointer,
        pointer_identity=lambda value, _binding: value,
        provider_capabilities=frozenset(
            {
                "native_handle": "group.native_handle.v1",
                "peer_ptr_table": "group.peer_ptr_table.v1",
            }[projection]
            for projection in pointers
        ),
        allocate_tensor=allocate_tensor,
        validate_allocation=validate_allocation,
        is_capturing=capture,
        execution_scope=scope,
        synchronize=synchronize,
    )


def _step(
    name: str,
    bindings: tuple[ArtifactBinding, ...],
    executor,
    *,
    plan: str = "default",
    step: int = 0,
    role: str = "run",
    specializes: tuple[tuple[str, object], ...] = (),
    prelaunch: tuple[ArtifactPrelaunch, ...] = (),
) -> ArtifactRuntimeStep:
    return ArtifactRuntimeStep(
        name=name,
        plan=plan,
        step=step,
        role=role,
        bindings=bindings,
        specializes=specializes,  # type: ignore[arg-type]
        prelaunch=prelaunch,
        executor=executor,
    )


def _blockscore_bindings() -> tuple[ArtifactBinding, ...]:
    return (
        ArtifactBinding(
            "input.q",
            "tensor",
            unsqueeze=(-1,),
            assumed_align=16,
            leading_dim=1,
        ),
        ArtifactBinding(
            "input.index_k",
            "tensor",
            unsqueeze=(-1,),
            assumed_align=16,
            leading_dim=1,
        ),
        ArtifactBinding(
            "output.block_scores",
            "tensor",
            assumed_align=4,
            leading_dim=1,
        ),
        ArtifactBinding("input.prefix_len", "scalar", cast="i32"),
        ArtifactBinding("input.scale", "scalar", cast="f32"),
        ArtifactBinding("stream.current", "stream"),
    )


def test_generic_blockscore_projection_has_no_slot_adapter() -> None:
    calls: list[tuple[object, ...]] = []
    entry = ArtifactRuntimeEntry(
        call_abi=MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
        steps=(
            _step(
                "blockscore",
                _blockscore_bindings(),
                lambda *args: calls.append(args),
                specializes=(("input.block_size", 128),),
                prelaunch=(
                    ArtifactPrelaunch("fill", "output.block_scores", "-inf"),
                ),
            ),
        ),
        provider=_provider(),
    )
    q = FakeTensor("q", (2, 128), pointer=16)
    index_k = FakeTensor("index_k", (132, 128), pointer=32)
    out = FakeTensor("out", (2, 2), stride=(9, 1), pointer=4)

    entry(q, index_k, 130, 0.125, 128, out)

    assert out.fills == [float("-inf")]
    assert len(calls) == 1
    assert calls[0][0][:2] == ("descriptor", q)
    assert calls[0][1][:2] == ("descriptor", index_k)
    assert calls[0][2][:2] == ("descriptor", out)
    assert calls[0][3:] == (130, 0.125, ("stream", 7))


def test_multi_step_plan_executes_in_sealed_step_order() -> None:
    observed: list[str] = []
    entry = ArtifactRuntimeEntry(
        call_abi=SILU_AND_MUL_CALL_ABI,
        steps=(
            _step(
                "second",
                (ArtifactBinding("output.out", "tensor"),),
                lambda *_args: observed.append("second"),
                step=2,
            ),
            _step(
                "first",
                (ArtifactBinding("input.x", "tensor"),),
                lambda *_args: observed.append("first"),
                step=1,
            ),
        ),
        provider=_provider(),
    )

    entry(FakeTensor("x", (2, 8)), FakeTensor("out", (2, 4)))
    assert observed == ["first", "second"]


def test_unique_most_specific_matching_plan_wins() -> None:
    observed: list[str] = []
    bindings = (ArtifactBinding("output.block_scores", "tensor"),)
    entry = ArtifactRuntimeEntry(
        call_abi=MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
        steps=(
            _step(
                "fallback",
                bindings,
                lambda *_args: observed.append("fallback"),
                plan="fallback",
            ),
            _step(
                "specialized",
                bindings,
                lambda *_args: observed.append("specialized"),
                plan="specialized",
                specializes=(("input.block_size", 128),),
            ),
        ),
        provider=_provider(),
    )
    q = FakeTensor("q", (2, 128))
    index_k = FakeTensor("index_k", (132, 128))

    entry(q, index_k, 130, 0.125, 128, FakeTensor("out128", (2, 2)))
    entry(q, index_k, 130, 0.125, 64, FakeTensor("out64", (2, 3)))
    assert observed == ["specialized", "fallback"]


def test_aggregate_projection_rejects_axis_outside_live_rank() -> None:
    aggregate = ArtifactBinding(
        None,
        "aggregate",
        cast="Shape",
        component_cast="i64",
        components=(
            ArtifactAggregateComponent(
                source="input.q", projection="shape", axis=2, cast="i64"
            ),
        ),
    )
    entry = ArtifactRuntimeEntry(
        call_abi=MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
        steps=(
            _step(
                "aggregate",
                (aggregate, ArtifactBinding("output.block_scores", "tensor")),
                lambda *_args: None,
            ),
        ),
        provider=_provider(),
    )
    with pytest.raises(ArtifactRuntimeError, match="outside live rank"):
        entry(
            FakeTensor("q", (2, 128)),
            FakeTensor("index_k", (132, 128)),
            130,
            0.125,
            128,
            FakeTensor("out", (2, 2)),
        )


def test_derived_scalar_cast_rejects_runtime_overflow() -> None:
    entry = ArtifactRuntimeEntry(
        call_abi=MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
        steps=(
            _step(
                "scalar",
                (
                    ArtifactBinding(
                        "input.q",
                        "scalar",
                        cast="u8",
                        projection="shape",
                        axis=0,
                    ),
                    ArtifactBinding("output.block_scores", "tensor"),
                ),
                lambda *_args: None,
            ),
        ),
        provider=_provider(),
    )
    with pytest.raises(ArtifactRuntimeError, match="outside u8 range"):
        entry(
            FakeTensor("q", (300, 128)),
            FakeTensor("index_k", (132, 128)),
            130,
            0.125,
            128,
            FakeTensor("out", (300, 2)),
        )


def test_alignment_assumption_is_checked_by_provider() -> None:
    entry = ArtifactRuntimeEntry(
        call_abi=SILU_AND_MUL_CALL_ABI,
        steps=(
            _step(
                "aligned",
                (
                    ArtifactBinding("input.x", "tensor", assumed_align=16),
                    ArtifactBinding("output.out", "tensor"),
                ),
                lambda *_args: None,
            ),
        ),
        provider=_provider(),
    )
    with pytest.raises(ArtifactRuntimeError, match="not 16-byte aligned"):
        entry(FakeTensor("x", (2, 8), pointer=3), FakeTensor("out", (2, 4)))


def test_group_native_capability_fails_closed_during_runtime_construction() -> None:
    with pytest.raises(
        ArtifactRuntimeError,
        match="lacks sealed capabilities.*group.native_handle.v1",
    ):
        ArtifactRuntimeEntry(
            call_abi=COLLECTIVE_ALL_REDUCE_CALL_ABI,
            steps=(
                _step(
                    "collective",
                    (
                        ArtifactBinding(
                            "group.current", "pointer", projection="native_handle"
                        ),
                        ArtifactBinding("output.out", "tensor"),
                    ),
                    lambda *_args: None,
                ),
            ),
            provider=_provider(),
        )


def test_group_peer_table_provider_receives_exact_sealed_local_buffer() -> None:
    observed: list[object] = []
    resource_plan = ArtifactResourcePlan(
        slot=COLLECTIVE_ALL_REDUCE_CALL_ABI.slot,
        resources=(
            ArtifactResource(
                name="state.ipc_data",
                dtype="u8",
                alignment=16,
                lifetime="engine",
                shape=(
                    ArtifactShapeExtent(
                        factors=(ArtifactShapeFactor(static=16),),
                    ),
                ),
                scope="group_ipc",
            ),
        ),
    )
    limits, budget = _runtime_authority()

    def peer_table(group, binding, local_buffer):
        observed.append((group, binding.peer_resource, local_buffer))
        return ("peers", local_buffer.data_ptr())

    calls: list[object] = []
    entry = ArtifactRuntimeEntry(
        call_abi=COLLECTIVE_ALL_REDUCE_CALL_ABI,
        steps=(
            _step(
                "collective",
                (
                    ArtifactBinding(
                        "group.current",
                        "pointer",
                        projection="peer_ptr_table",
                        peer_resource="state.ipc_data",
                    ),
                    ArtifactBinding("output.out", "tensor"),
                ),
                lambda peers, _out: calls.append(peers),
            ),
        ),
        provider=_provider(group_pointers={"peer_ptr_table": peer_table}),
        artifact_resources=resource_plan,
        limits=limits,
        allocation_budget=budget,
    )
    group = FakeGroup()

    entry(
        FakeTensor("x", (2, 8)),
        FakeTensor("out", (2, 8)),
        group,
    )

    assert len(observed) == 1
    assert observed[0][0] is group
    assert observed[0][1] == "state.ipc_data"
    assert observed[0][2].name.startswith("u8:")
    assert calls == [("peers", observed[0][2].data_ptr())]


def test_multiple_nonpositional_groups_derive_their_own_rank_size_facts() -> None:
    call_abi = SlotCallABI(
        slot="test.multi_group",
        resources=(
            SlotResource("input.x", "tensor"),
            SlotResource("output.out", "tensor", access="write"),
            SlotResource("group.g000", "group"),
            SlotResource(
                "group.g000_rank",
                "scalar",
                scalar_type="i32",
                capability_field="rank",
                provider_resource="group.g000",
                provider_projection="rank",
            ),
            SlotResource("group.g001", "group"),
            SlotResource(
                "group.g001_size",
                "scalar",
                scalar_type="i32",
                capability_field="world_size",
                provider_resource="group.g001",
                provider_projection="size",
            ),
        ),
        call_args=("input.x", "output.out"),
    )
    observed: list[tuple[int, int]] = []
    entry = ArtifactRuntimeEntry(
        call_abi=call_abi,
        steps=(
            _step(
                "run",
                (
                    ArtifactBinding("group.g000_rank", "scalar", cast="i32"),
                    ArtifactBinding("group.g001_size", "scalar", cast="i32"),
                    ArtifactBinding("output.out", "tensor"),
                ),
                lambda rank, size, _out: observed.append((rank, size)),
            ),
        ),
        provider=_provider(),
    )

    with artifact_call_resources(
        {
            "group.g000": FakeGroup(rank_value=1, size_value=4),
            "group.g001": FakeGroup(rank_value=2, size_value=8),
        }
    ):
        entry(FakeTensor("x", (2, 8)), FakeTensor("out", (2, 8)))

    assert observed == [(1, 8)]


def _resource(name: str, lifetime: str) -> ArtifactResource:
    return ArtifactResource(
        name=name,
        dtype="u8",
        alignment=16,
        lifetime=lifetime,
        shape=(
            ArtifactShapeExtent(
                factors=(ArtifactShapeFactor(static=16),),
            ),
        ),
    )


def _runtime_authority(max_bytes: int = 1 << 20):
    limits = ArtifactRuntimeLimits(max_live_bytes=max_bytes, max_allocation_keys=32)
    return limits, ArtifactAllocationBudget(limits)


def test_lifecycle_uses_validator_storage_and_close_destroys_once() -> None:
    observed: list[str] = []
    resource_plan = ArtifactResourcePlan(
        slot=SILU_AND_MUL_CALL_ABI.slot,
        resources=(
            _resource("state.counter", "engine"),
            _resource("prepared.lookup", "prepared"),
        ),
    )
    limits, budget = _runtime_authority()
    entry = ArtifactRuntimeEntry(
        call_abi=SILU_AND_MUL_CALL_ABI,
        steps=(
            _step(
                "init",
                (ArtifactBinding("state.counter", "tensor"),),
                lambda *_args: observed.append("init"),
                step=0,
                role="init",
            ),
            _step(
                "prepare",
                (ArtifactBinding("prepared.lookup", "tensor"),),
                lambda *_args: observed.append("prepare"),
                step=1,
                role="prepare",
            ),
            _step(
                "run",
                (
                    ArtifactBinding("prepared.lookup", "tensor"),
                    ArtifactBinding("state.counter", "tensor"),
                    ArtifactBinding("output.out", "tensor"),
                ),
                lambda *_args: observed.append("run"),
                step=2,
                role="run",
            ),
            _step(
                "destroy_prepared",
                (ArtifactBinding("prepared.lookup", "tensor"),),
                lambda *_args: observed.append("destroy_prepared"),
                step=3,
                role="destroy",
            ),
            _step(
                "destroy_state",
                (ArtifactBinding("state.counter", "tensor"),),
                lambda *_args: observed.append("destroy_state"),
                step=4,
                role="destroy",
            ),
        ),
        provider=_provider(),
        artifact_resources=resource_plan,
        limits=limits,
        allocation_budget=budget,
    )

    entry(FakeTensor("x", (2, 8)), FakeTensor("out1", (2, 4)))
    entry(FakeTensor("x2", (2, 8)), FakeTensor("out2", (2, 4)))
    assert observed == ["init", "prepare", "run", "run"]
    # Each 16-byte payload retains an aligned backing allocation with up to
    # 15 bytes of padding, and each buffer consumes its own allocation key.
    assert entry.allocated_bytes == 62
    assert entry.allocated_keys == 2
    assert budget.allocated_keys == 2
    entry.close()
    entry.close()
    assert observed[-2:] == ["destroy_prepared", "destroy_state"]
    assert budget.allocated_bytes == 0
    with pytest.raises(ArtifactRuntimeError, match="closed"):
        entry(FakeTensor("x", (2, 8)), FakeTensor("late", (2, 4)))


def test_result_free_model_prepare_args_can_prepare_implicitly_before_run() -> None:
    observed: list[tuple[str, tuple[int, ...]]] = []
    call_abi = SlotCallABI(
        slot="test.model_prepare",
        resources=(
            SlotResource("input.weight", "tensor"),
            SlotResource("input.x", "tensor"),
            SlotResource("output.out", "tensor", access="write"),
        ),
        call_args=("input.weight", "input.x", "output.out"),
        prepare_args=("input.weight",),
    )
    resource_plan = ArtifactResourcePlan(
        slot=call_abi.slot,
        resources=(
            ArtifactResource(
                name="prepared.lookup",
                dtype="u8",
                alignment=16,
                lifetime="prepared",
                shape=(
                    ArtifactShapeExtent(
                        factors=(
                            ArtifactShapeFactor(
                                source="input.weight",
                                projection="shape",
                                axis=1,
                                upper_bound=4096,
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    limits, budget = _runtime_authority()
    entry = ArtifactRuntimeEntry(
        call_abi=call_abi,
        steps=(
            _step(
                "prepare",
                (
                    ArtifactBinding("input.weight", "tensor"),
                    ArtifactBinding("prepared.lookup", "tensor"),
                ),
                lambda _x, prepared: observed.append(("prepare", prepared[1].shape)),
                role="prepare",
            ),
            _step(
                "run",
                (
                    ArtifactBinding("prepared.lookup", "tensor"),
                    ArtifactBinding("output.out", "tensor"),
                ),
                lambda prepared, _out: observed.append(("run", prepared[1].shape)),
                step=1,
            ),
        ),
        provider=_provider(),
        artifact_resources=resource_plan,
        limits=limits,
        allocation_budget=budget,
    )

    weight = FakeTensor("weight", (2, 8))
    entry(weight, FakeTensor("x", (2, 8)), FakeTensor("out", (2, 4)))
    entry(weight, FakeTensor("x2", (2, 8)), FakeTensor("out2", (2, 4)))

    assert observed == [
        ("prepare", (8,)),
        ("run", (8,)),
        ("run", (8,)),
    ]


def test_validator_prepare_frame_flows_into_moe_run_and_rejects_foreign_state() -> None:
    resource_plan = ArtifactResourcePlan(
        slot=MOE_FUSED_EXPERTS_CALL_ABI.slot,
        resources=(_resource("prepared.packed_weights", "prepared"),),
    )
    limits, budget = _runtime_authority()
    observed: list[str] = []
    entry = ArtifactRuntimeEntry(
        call_abi=MOE_FUSED_EXPERTS_CALL_ABI,
        steps=(
            _step(
                "prepare",
                (
                    ArtifactBinding("input.w13", "tensor"),
                    ArtifactBinding("input.w2", "tensor"),
                    ArtifactBinding("prepared.packed_weights", "tensor"),
                ),
                lambda *_args: observed.append("prepare"),
                step=0,
                role="prepare",
            ),
            _step(
                "run",
                (
                    ArtifactBinding("prepared.packed_weights", "tensor"),
                    ArtifactBinding("output.out", "tensor"),
                ),
                lambda *_args: observed.append("run"),
                step=1,
            ),
        ),
        provider=_provider(),
        artifact_resources=resource_plan,
        limits=limits,
        allocation_budget=budget,
    )
    prepared = entry.prepare(
        FakeTensor("w13", (8, 8)), FakeTensor("w2", (8, 8))
    )
    assert type(prepared) is ArtifactPreparedState
    entry(
        FakeTensor("x", (2, 8)),
        FakeTensor("ids", (2, 2)),
        FakeTensor("weights", (2, 2)),
        prepared,
        FakeTensor("out", (2, 8)),
    )
    assert observed == ["prepare", "run"]
    with pytest.raises(ArtifactRuntimeError, match="foreign prepared-state"):
        entry(
            FakeTensor("x", (2, 8)),
            FakeTensor("ids", (2, 2)),
            FakeTensor("weights", (2, 2)),
            ArtifactPreparedState(object(), (), object()),
            FakeTensor("out", (2, 8)),
        )


def test_missing_workspace_allocation_fails_during_capture_then_reuses_warmup() -> None:
    capturing = {"value": True}
    resource_plan = ArtifactResourcePlan(
        slot=SILU_AND_MUL_CALL_ABI.slot,
        resources=(_resource("workspace.scratch", "call"),),
    )
    limits, budget = _runtime_authority()
    entry = ArtifactRuntimeEntry(
        call_abi=SILU_AND_MUL_CALL_ABI,
        steps=(
            _step(
                "run",
                (
                    ArtifactBinding("workspace.scratch", "tensor"),
                    ArtifactBinding("output.out", "tensor"),
                ),
                lambda *_args: None,
            ),
        ),
        provider=_provider(capture=lambda: capturing["value"]),
        artifact_resources=resource_plan,
        limits=limits,
        allocation_budget=budget,
    )
    args = (FakeTensor("x", (2, 8)), FakeTensor("out", (2, 4)))
    with pytest.raises(ArtifactRuntimeError, match="before CUDA graph capture"):
        entry(*args)
    capturing["value"] = False
    entry(*args)
    capturing["value"] = True
    entry(*args)


def test_workspace_frames_are_isolated_by_execution_scope() -> None:
    live_scope = {"value": ("fake-cuda", 0, 7)}
    seen: list[int] = []
    resource_plan = ArtifactResourcePlan(
        slot=SILU_AND_MUL_CALL_ABI.slot,
        resources=(_resource("workspace.scratch", "call"),),
    )
    limits, budget = _runtime_authority()
    entry = ArtifactRuntimeEntry(
        call_abi=SILU_AND_MUL_CALL_ABI,
        steps=(
            _step(
                "run",
                (
                    ArtifactBinding("workspace.scratch", "pointer", projection="device_ptr"),
                    ArtifactBinding("output.out", "tensor"),
                ),
                lambda pointer, _out: seen.append(pointer[1]),
            ),
        ),
        provider=_provider(scope=lambda: live_scope["value"]),
        artifact_resources=resource_plan,
        limits=limits,
        allocation_budget=budget,
    )
    args = (FakeTensor("x", (2, 8)), FakeTensor("out", (2, 4)))

    entry(*args)
    live_scope["value"] = ("fake-cuda", 0, 8)
    entry(*args)
    live_scope["value"] = ("fake-cuda", 0, 7)
    entry(*args)

    assert seen[0] != seen[1]
    assert seen[0] == seen[2]
    assert entry.allocated_keys == 2


def test_alignment_padding_and_each_buffer_are_charged_to_shared_budget() -> None:
    resource_plan = ArtifactResourcePlan(
        slot=SILU_AND_MUL_CALL_ABI.slot,
        resources=(
            ArtifactResource(
                name="workspace.first",
                dtype="u8",
                alignment=1024,
                lifetime="call",
                shape=(
                    ArtifactShapeExtent(
                        factors=(ArtifactShapeFactor(static=1),),
                    ),
                ),
            ),
            ArtifactResource(
                name="workspace.second",
                dtype="u8",
                alignment=1024,
                lifetime="call",
                shape=(
                    ArtifactShapeExtent(
                        factors=(ArtifactShapeFactor(static=1),),
                    ),
                ),
            ),
        ),
    )
    limits = ArtifactRuntimeLimits(max_live_bytes=2047, max_allocation_keys=8)
    budget = ArtifactAllocationBudget(limits)
    entry = ArtifactRuntimeEntry(
        call_abi=SILU_AND_MUL_CALL_ABI,
        steps=(
            _step(
                "run",
                (
                    ArtifactBinding("workspace.first", "tensor"),
                    ArtifactBinding("workspace.second", "tensor"),
                    ArtifactBinding("output.out", "tensor"),
                ),
                lambda *_args: None,
            ),
        ),
        provider=_provider(),
        artifact_resources=resource_plan,
        limits=limits,
        allocation_budget=budget,
    )

    with pytest.raises(ArtifactRuntimeError, match="live-byte budget"):
        entry(FakeTensor("x", (2, 8)), FakeTensor("out", (2, 4)))
    assert budget.allocated_bytes == 0
    assert budget.allocated_keys == 0


def test_prepared_token_does_not_retain_frames_after_close() -> None:
    resource_plan = ArtifactResourcePlan(
        slot=MOE_FUSED_EXPERTS_CALL_ABI.slot,
        resources=(_resource("prepared.packed_weights", "prepared"),),
    )
    limits, budget = _runtime_authority()
    entry = ArtifactRuntimeEntry(
        call_abi=MOE_FUSED_EXPERTS_CALL_ABI,
        steps=(
            _step(
                "prepare",
                (
                    ArtifactBinding("input.w13", "tensor"),
                    ArtifactBinding("prepared.packed_weights", "tensor"),
                ),
                lambda *_args: None,
                role="prepare",
            ),
            _step(
                "run",
                (
                    ArtifactBinding("prepared.packed_weights", "tensor"),
                    ArtifactBinding("output.out", "tensor"),
                ),
                lambda *_args: None,
                step=1,
            ),
        ),
        provider=_provider(),
        artifact_resources=resource_plan,
        limits=limits,
        allocation_budget=budget,
    )
    prepared = entry.prepare(
        FakeTensor("w13", (8, 8)), FakeTensor("w2", (8, 8))
    )
    assert not hasattr(prepared, "_frames")
    assert budget.allocated_bytes > 0

    entry.close()

    assert budget.allocated_bytes == 0
    assert budget.allocated_keys == 0
    assert prepared._token is not None
