import pytest

from optima.artifact_abi import (
    ArtifactAggregateComponent,
    ArtifactABIError,
    ArtifactBinding,
    ArtifactPrelaunch,
    MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
    ProviderCapabilityRequirement,
    SpecializationCapabilityRequirement,
    SLOT_CALL_ABIS,
    SlotCallABI,
    SlotResource,
    checked_integer_cast,
    parse_artifact_bindings,
    parse_provider_capability_requirements,
    parse_specialization_capability_requirements,
)


_EXPECTED_CALL_ARGS = {
    "activation.silu_and_mul": ("input.x", "output.out"),
    "norm.rmsnorm": ("input.x", "input.weight", "output.out", "input.eps"),
    "attention.sdpa": (
        "input.q", "input.k", "input.v", "output.out", "input.sm_scale", "input.causal"
    ),
    "attention.decode": (
        "input.q", "input.k", "input.v", "input.seq_lens", "input.sm_scale", "output.out"
    ),
    "moe.fused_experts": (
        "input.x", "input.topk_ids", "input.topk_weights", "prepared.state", "output.out"
    ),
    "collective.all_reduce": ("input.x", "output.out", "group.current"),
    "collective.ar_residual_rmsnorm": (
        "input.x", "input.residual", "input.weight", "input.eps",
        "output.out_norm", "output.out_residual", "group.current",
    ),
    "collective.moe_finalize_ar_rmsnorm": (
        "input.gemm_out", "input.row_map", "input.scales", "input.residual",
        "input.weight", "input.eps", "output.out_norm", "output.out_residual",
        "group.current",
    ),
    "moe.fused_experts_reduce": (
        "input.x", "input.topk_ids", "input.topk_weights", "prepared.state",
        "output.out", "group.current",
    ),
    "attention.msa_block_score": (
        "input.q", "input.index_k", "input.seq_lens", "input.block_size",
        "output.block_scores",
    ),
    "attention.msa_prefill_block_score": (
        "input.q", "input.index_k", "input.prefix_len", "input.scale",
        "input.block_size", "output.block_scores",
    ),
}


def _blockscore_bindings():
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


def test_every_slot_has_one_shared_validator_owned_call_abi():
    from optima.slots import SLOTS

    assert set(SLOT_CALL_ABIS) == set(SLOTS) == set(_EXPECTED_CALL_ARGS)
    for slot_name, call_args in _EXPECTED_CALL_ARGS.items():
        abi = SLOT_CALL_ABIS[slot_name]
        assert abi.call_args == call_args
        assert SLOTS[slot_name].call_abi is abi


def test_prepare_and_collective_boundaries_are_explicit_not_folded_into_run():
    for slot_name in ("moe.fused_experts", "moe.fused_experts_reduce"):
        abi = SLOT_CALL_ABIS[slot_name]
        assert abi.prepare_args == ("input.w13", "input.w2")
        assert abi.prepare_result == "prepared.state"
        assert "input.w13" not in abi.call_args
        assert "input.w2" not in abi.call_args

    for slot_name in (
        "collective.all_reduce",
        "collective.ar_residual_rmsnorm",
        "collective.moe_finalize_ar_rmsnorm",
        "moe.fused_experts_reduce",
    ):
        abi = SLOT_CALL_ABIS[slot_name]
        group = abi.by_name["group.current"]
        assert group.provider_capabilities == (
            "group.native_handle.v1",
            "group.peer_ptr_table.v1",
        )
        assert abi.by_name["group.rank"].capability_field == "rank"
        assert abi.by_name["group.size"].capability_field == "world_size"


def test_slot_call_abi_accepts_declarative_blockscore_projection():
    assert MSA_PREFILL_BLOCK_SCORE_CALL_ABI.call_args == (
        "input.q",
        "input.index_k",
        "input.prefix_len",
        "input.scale",
        "input.block_size",
        "output.block_scores",
    )
    specializes = MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
        role="run",
        bindings=_blockscore_bindings(),
        specializes={"input.block_size": 128},
        prelaunch=(
            ArtifactPrelaunch("fill", "output.block_scores", "-inf"),
        ),
    )

    assert specializes == (("input.block_size", 128),)
    assert MSA_PREFILL_BLOCK_SCORE_CALL_ABI.specialization_capabilities(
        specializes
    ) == {"block_size": 128}


def test_slot_call_abi_rejects_unknown_resource_and_kind_confusion():
    bindings = list(_blockscore_bindings())
    bindings[0] = ArtifactBinding("input.not_declared", "tensor")
    with pytest.raises(ArtifactABIError, match="unknown slot resource"):
        MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
            role="run", bindings=bindings, specializes={}, prelaunch=()
        )

    bindings[0] = ArtifactBinding("input.q", "scalar", cast="i64")
    with pytest.raises(ArtifactABIError, match="not allowed for slot resource"):
        MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
            role="run", bindings=bindings, specializes={}, prelaunch=()
        )


def test_tensor_projection_matrix_covers_descriptor_pointer_and_metadata():
    bindings = (
        ArtifactBinding(
            "input.q", "pointer", projection="device_ptr", assumed_align=16
        ),
        ArtifactBinding("input.q", "scalar", cast="i32", projection="shape", axis=0),
        ArtifactBinding("input.q", "scalar", cast="i64", projection="stride", axis=1),
        ArtifactBinding("input.q", "scalar", cast="u64", projection="numel"),
        ArtifactBinding("input.q", "scalar", cast="i32", projection="rank"),
        ArtifactBinding(
            "input.q", "scalar", cast="i64", projection="storage_offset"
        ),
        ArtifactBinding("output.block_scores", "tensor"),
    )
    MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
        role="run", bindings=bindings, specializes={}, prelaunch=()
    )

    assert bindings[0].native_projection == ("pointer", None)
    assert bindings[1].native_projection == ("scalar", "i32")
    assert bindings[1].integer_range == (-(1 << 31), (1 << 31) - 1)


def test_tensor_projection_matrix_rejects_implicit_or_ill_typed_coercions():
    with pytest.raises(ArtifactABIError, match="not allowed for slot resource"):
        MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
            role="run",
            bindings=(
                ArtifactBinding("input.q", "pointer"),
                ArtifactBinding("output.block_scores", "tensor"),
            ),
            specializes={},
            prelaunch=(),
        )
    with pytest.raises(ArtifactABIError, match="requires an integer cast"):
        ArtifactBinding("input.q", "scalar", cast="f32", projection="shape", axis=0)
    with pytest.raises(ArtifactABIError, match="axis must be"):
        ArtifactBinding("input.q", "scalar", cast="i64", projection="shape")
    with pytest.raises(ArtifactABIError, match="may not declare an axis"):
        ArtifactBinding("input.q", "scalar", cast="i64", projection="numel", axis=0)


def test_output_metadata_projection_does_not_claim_write_coverage():
    with pytest.raises(ArtifactABIError, match="does not write slot outputs"):
        MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
            role="run",
            bindings=(
                ArtifactBinding(
                    "output.block_scores",
                    "scalar",
                    cast="i64",
                    projection="shape",
                    axis=0,
                ),
            ),
            specializes={},
            prelaunch=(),
        )


def test_integer_projection_cast_is_checked_not_truncated():
    assert checked_integer_cast(255, "u8", field="shape") == 255
    with pytest.raises(ArtifactABIError, match="outside u8 range"):
        checked_integer_cast(256, "u8", field="shape")
    with pytest.raises(ArtifactABIError, match="must resolve to an integer"):
        checked_integer_cast(True, "i32", field="rank")


def test_bounded_nested_aggregate_projection_covers_cute_algebra_values():
    aggregate = ArtifactBinding(
        None,
        "aggregate",
        cast="Tile",
        component_cast="i64",
        components=(
            ArtifactAggregateComponent(
                source="input.q", projection="shape", axis=0, cast="i64"
            ),
            (
                ArtifactAggregateComponent(static=128),
                ArtifactAggregateComponent(
                    source="input.q", projection="stride", axis=1, cast="i64"
                ),
                ArtifactAggregateComponent(
                    source="input.prefix_len", projection="value", cast="i64"
                ),
            ),
        ),
    )
    MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
        role="run",
        bindings=(aggregate, ArtifactBinding("output.block_scores", "tensor")),
        specializes={},
        prelaunch=(),
    )
    assert aggregate.dynamic_arity == 3
    assert aggregate.native_projection == ("aggregate", "i64", 3)
    assert aggregate.to_dict()["components"][1][0] == {"static": 128}


def test_aggregate_projection_rejects_cast_mismatch_and_arbitrary_components():
    with pytest.raises(ArtifactABIError, match="must equal component_cast"):
        ArtifactBinding(
            None,
            "aggregate",
            cast="Shape",
            component_cast="i32",
            components=(
                ArtifactAggregateComponent(
                    source="input.q", projection="shape", axis=0, cast="i64"
                ),
            ),
        )
    with pytest.raises(ArtifactABIError, match="unknown=.*callback"):
        parse_artifact_bindings(
            [
                {
                    "kind": "aggregate",
                    "cast": "Shape",
                    "component_cast": "i32",
                    "components": [
                        {
                            "source": "input.q",
                            "projection": "shape",
                            "axis": 0,
                            "cast": "i32",
                            "callback": "miner.code",
                        }
                    ],
                }
            ],
            field="bindings",
        )


def test_slot_call_abi_requires_every_output_to_be_written():
    bindings = tuple(
        binding
        for binding in _blockscore_bindings()
        if binding.source != "output.block_scores"
    )
    with pytest.raises(ArtifactABIError, match="does not write slot outputs"):
        MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
            role="run", bindings=bindings, specializes={}, prelaunch=()
        )


def test_multi_export_pipeline_checks_output_coverage_across_run_steps():
    first = tuple(
        binding
        for binding in _blockscore_bindings()
        if binding.source != "output.block_scores"
    )
    MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
        role="run",
        bindings=first,
        specializes={"input.block_size": 128},
        prelaunch=(),
        require_outputs=False,
    )
    MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_pipeline(
        (
            ("run", first, ()),
            (
                "run",
                (ArtifactBinding("output.block_scores", "tensor"),),
                (),
            ),
        )
    )
    with pytest.raises(ArtifactABIError, match="does not write slot outputs"):
        MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_pipeline(
            (("run", first, ()),)
        )


def test_compile_time_specialization_requires_validator_descriptor_fact():
    with pytest.raises(ArtifactABIError, match="lacks a validator capability fact"):
        MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
            role="run",
            bindings=_blockscore_bindings(),
            specializes={"input.prefix_len": 0},
            prelaunch=(),
        )


def test_specialization_capability_is_policy_and_used_mapping_is_sealed():
    common = (
        SlotResource("input.x", "tensor"),
        SlotResource("input.tile", "scalar", scalar_type="i32"),
        SlotResource("output.y", "tensor", access="write"),
    )
    unavailable = SlotCallABI(
        slot="activation.specialized",
        resources=common,
        call_args=("input.x", "input.tile", "output.y"),
    )
    available = SlotCallABI(
        slot="activation.specialized",
        resources=(
            common[0],
            SlotResource(
                "input.tile",
                "scalar",
                scalar_type="i32",
                capability_field="tile_size",
            ),
            common[2],
        ),
        call_args=unavailable.call_args,
    )

    assert unavailable.to_dict() != available.to_dict()
    assert unavailable.identity_snapshot() == available.identity_snapshot()
    with pytest.raises(ArtifactABIError, match="lacks a validator capability"):
        unavailable.validate_plan(
            role="run",
            bindings=(ArtifactBinding("output.y", "tensor"),),
            specializes={"input.tile": 128},
            prelaunch=(),
        )
    specializes = available.validate_plan(
        role="run",
        bindings=(ArtifactBinding("output.y", "tensor"),),
        specializes={"input.tile": 128},
        prelaunch=(),
    )
    requirements = available.specialization_capability_requirements(specializes)
    assert requirements == (
        SpecializationCapabilityRequirement(
            source="input.tile",
            capability_field="tile_size",
        ),
    )
    remapped = parse_specialization_capability_requirements(
        [
            {
                "capability_field": "tile_extent",
                "source": "input.tile",
            }
        ],
        field="requirements",
    )
    assert remapped != requirements


def test_resource_model_is_provider_neutral_for_state_workspace_and_groups():
    abi = SlotCallABI(
        slot="collective.future",
        resources=(
            SlotResource("input.x", "tensor"),
            SlotResource("output.y", "tensor", access="write"),
            SlotResource("workspace.scratch", "tensor", access="readwrite"),
            SlotResource("state.counter", "tensor", access="readwrite"),
            SlotResource("group.rank", "scalar", scalar_type="i32"),
            SlotResource("group.handle", "group"),
            SlotResource("stream.current", "stream"),
        ),
        call_args=("input.x", "output.y", "group.handle"),
    )
    bindings = (
        ArtifactBinding("input.x", "tensor", assumed_align=16),
        ArtifactBinding("output.y", "tensor", assumed_align=16),
        ArtifactBinding("workspace.scratch", "tensor"),
        ArtifactBinding("state.counter", "tensor"),
        ArtifactBinding("group.rank", "scalar", cast="i32"),
        ArtifactBinding("group.handle", "group"),
        ArtifactBinding("stream.current", "stream"),
    )
    assert abi.validate_plan(
        role="run", bindings=bindings, specializes={}, prelaunch=()
    ) == ()


def test_group_projections_are_closed_and_provider_capability_gated():
    resources = (
        SlotResource("input.x", "tensor"),
        SlotResource("output.y", "tensor", access="write"),
        SlotResource(
            "group.current",
            "group",
            provider_capabilities=("group.native_handle.v1",),
        ),
    )
    abi = SlotCallABI(
        slot="collective.projected",
        resources=resources,
        call_args=("input.x", "output.y", "group.current"),
    )
    bindings = (
        ArtifactBinding("output.y", "tensor"),
        ArtifactBinding("group.current", "scalar", cast="i32", projection="rank"),
        ArtifactBinding("group.current", "scalar", cast="i64", projection="size"),
        ArtifactBinding(
            "group.current", "pointer", projection="native_handle"
        ),
    )
    abi.validate_plan(
        role="run",
        bindings=bindings,
        specializes={},
        prelaunch=(),
    )
    requirements = abi.provider_capability_requirements(bindings)
    assert requirements == (
        ProviderCapabilityRequirement(
            source="group.current",
            projection="native_handle",
            capability="group.native_handle.v1",
        ),
    )

    expanded = SlotCallABI(
        slot=abi.slot,
        resources=(
            resources[0],
            resources[1],
            SlotResource(
                "group.current",
                "group",
                provider_capabilities=(
                    "group.native_handle.v1",
                    "group.peer_ptr_table.v1",
                ),
            ),
        ),
        call_args=abi.call_args,
    )
    assert abi.to_dict() != expanded.to_dict()
    assert abi.identity_snapshot() == expanded.identity_snapshot()
    assert expanded.provider_capability_requirements(bindings) == requirements

    sealed_new_version = parse_provider_capability_requirements(
        [
            {
                "capability": "group.native_handle.v2",
                "projection": "native_handle",
                "source": "group.current",
            }
        ],
        field="requirements",
    )
    assert sealed_new_version != requirements

    without_handle = SlotCallABI(
        slot="collective.no_handle",
        resources=(resources[0], resources[1], SlotResource("group.current", "group")),
        call_args=("input.x", "output.y", "group.current"),
    )
    with pytest.raises(ArtifactABIError, match="requires validator capability"):
        without_handle.validate_plan(
            role="run",
            bindings=(
                ArtifactBinding("output.y", "tensor"),
                ArtifactBinding(
                    "group.current", "pointer", projection="native_handle"
                ),
            ),
            specializes={},
            prelaunch=(),
        )


def test_provider_capability_requirement_parser_is_closed_and_canonical():
    with pytest.raises(ArtifactABIError, match="sorted canonically"):
        parse_provider_capability_requirements(
            [
                {
                    "capability": "group.peer_ptr_table.v1",
                    "projection": "peer_ptr_table",
                    "source": "group.current",
                },
                {
                    "capability": "group.native_handle.v1",
                    "projection": "native_handle",
                    "source": "group.current",
                },
            ],
            field="requirements",
        )
    with pytest.raises(ArtifactABIError, match="exactly"):
        parse_provider_capability_requirements(
            [
                {
                    "capability": "group.native_handle.v1",
                    "projection": "native_handle",
                    "source": "group.current",
                    "callback": "miner.adapter",
                }
            ],
            field="requirements",
        )


def test_binding_parser_is_closed_to_unknown_runtime_operations():
    with pytest.raises(ArtifactABIError, match="unknown=.*python_callback"):
        parse_artifact_bindings(
            [
                {
                    "source": "input.q",
                    "kind": "tensor",
                    "python_callback": "candidate.adapter",
                }
            ],
            field="bindings",
        )
