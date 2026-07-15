from __future__ import annotations

import pytest

from optima.artifact_abi import (
    ArtifactABIError,
    ArtifactBinding,
    ArtifactResource,
    ArtifactResourcePlan,
    ArtifactShapeExtent,
    ArtifactShapeFactor,
    COLLECTIVE_ALL_REDUCE_CALL_ABI,
    MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
    MOE_FUSED_EXPERTS_CALL_ABI,
    SlotCallABI,
    SlotResource,
    parse_artifact_bindings,
    parse_artifact_resources,
)
from optima.manifest import ManifestError, load_manifest


def _static_resource(
    name: str,
    *,
    lifetime: str,
    dtype: str = "u8",
    extent: int = 1,
    scope: str = "rank_local",
) -> ArtifactResource:
    return ArtifactResource(
        name=name,
        dtype=dtype,
        alignment=16,
        lifetime=lifetime,
        shape=(
            ArtifactShapeExtent(
                factors=(ArtifactShapeFactor(static=extent),),
            ),
        ),
        scope=scope,
    )


def _resource_rows():
    return (
        {
            "name": "workspace.scratch",
            "dtype": "f32",
            "alignment": 16,
            "lifetime": "call",
            "shape": (
                {
                    "factors": (
                        {
                            "source": "input.q",
                            "projection": "shape",
                            "axis": 0,
                            "upper_bound": 4096,
                        },
                        {"static": 4},
                    ),
                    "divisor": 2,
                },
            ),
        },
        {
            "name": "state.epoch",
            "dtype": "i32",
            "alignment": 16,
            "lifetime": "engine",
            "shape": ({"factors": ({"static": 1},)},),
        },
        {
            "name": "prepared.lookup",
            "dtype": "u8",
            "alignment": 128,
            "lifetime": "prepared",
            "shape": (
                {
                    "factors": (
                        {
                            "source": "input.q",
                            "projection": "numel",
                            "upper_bound": 1 << 20,
                        },
                    ),
                    "divisor": 256,
                },
            ),
        },
    )


def test_slot_authorizes_bounded_artifact_storage_without_per_slot_names():
    plan = parse_artifact_resources(
        _resource_rows(),
        call_abi=MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
        field="artifact_resources",
    )

    assert tuple(plan.by_name) == (
        "workspace.scratch",
        "state.epoch",
        "prepared.lookup",
    )
    assert plan.by_name["workspace.scratch"].max_bytes == 32_768
    assert plan.by_name["prepared.lookup"].max_bytes == 4096
    assert set(MSA_PREFILL_BLOCK_SCORE_CALL_ABI.resource_table(plan)) >= {
        "input.q",
        "output.block_scores",
        "workspace.scratch",
        "state.epoch",
        "prepared.lookup",
    }

    prepare = (ArtifactBinding("prepared.lookup", "tensor"),)
    run = (
        ArtifactBinding("output.block_scores", "tensor"),
        ArtifactBinding("prepared.lookup", "tensor"),
        ArtifactBinding("workspace.scratch", "tensor"),
        ArtifactBinding("state.epoch", "tensor"),
    )
    MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
        role="prepare",
        bindings=prepare,
        specializes={},
        prelaunch=(),
        require_outputs=False,
        artifact_resources=plan,
    )
    MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
        role="run",
        bindings=run,
        specializes={},
        prelaunch=(),
        artifact_resources=plan,
    )
    MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_pipeline(
        (("prepare", prepare, ()), ("run", run, ())),
        artifact_resources=plan,
    )
    plan.validate_pipeline(
        (("prepare", prepare, ()), ("run", run, ())),
        require_all=True,
    )


def test_bindings_may_name_only_declared_generated_resources():
    plan = parse_artifact_resources(
        (_resource_rows()[0],),
        call_abi=MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
        field="artifact_resources",
    )
    with pytest.raises(ArtifactABIError, match="unknown slot resource"):
        MSA_PREFILL_BLOCK_SCORE_CALL_ABI.validate_plan(
            role="run",
            bindings=(
                ArtifactBinding("output.block_scores", "tensor"),
                ArtifactBinding("workspace.not_declared", "tensor"),
            ),
            specializes={},
            prelaunch=(),
            artifact_resources=plan,
        )

def test_artifact_resource_scope_defaults_rank_local_and_seals_canonically():
    plan = parse_artifact_resources(
        (_resource_rows()[1],),
        call_abi=MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
        field="artifact_resources",
    )

    resource = plan.by_name["state.epoch"]
    assert resource.scope == "rank_local"
    assert "scope" not in resource.to_dict()

    group = _static_resource(
        "state.group_buffer", lifetime="engine", scope="group_ipc"
    )
    assert group.to_dict()["scope"] == "group_ipc"


@pytest.mark.parametrize(
    "row, message",
    [
        (
            {**_resource_rows()[1], "scope": "candidate_allocator"},
            "unsupported allocation scope",
        ),
        (
            {**_resource_rows()[0], "scope": "group_ipc"},
            "must have a persistent",
        ),
    ],
)
def test_artifact_resource_scope_is_closed_and_group_ipc_is_persistent(row, message):
    with pytest.raises(ArtifactABIError, match=message):
        parse_artifact_resources(
            (row,),
            call_abi=COLLECTIVE_ALL_REDUCE_CALL_ABI,
            field="artifact_resources",
        )


def test_prepared_shape_sources_exist_at_explicit_or_implicit_prepare_boundary():
    # MSA has no separate prepare_args boundary, so its validator-owned implicit
    # pre-run warmup can derive the prepared allocation from call_args.
    implicit = parse_artifact_resources(
        (_resource_rows()[2],),
        call_abi=MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
        field="artifact_resources",
    )
    assert implicit.by_name["prepared.lookup"].lifetime == "prepared"

    run_only = {
        **_resource_rows()[2],
        "shape": (
            {
                "factors": (
                    {
                        "source": "input.x",
                        "projection": "numel",
                        "upper_bound": 1 << 20,
                    },
                ),
            },
        ),
    }
    with pytest.raises(ArtifactABIError, match="unavailable at the validator prepare"):
        parse_artifact_resources(
            (run_only,),
            call_abi=MOE_FUSED_EXPERTS_CALL_ABI,
            field="artifact_resources",
        )


@pytest.mark.parametrize(
    "resources",
    [
        (
            _static_resource("state.engine_buffer", lifetime="engine"),
            _static_resource("prepared.weight_buffer", lifetime="prepared"),
        ),
        (
            _static_resource("state.rank_buffer", lifetime="engine"),
            _static_resource(
                "state.group_buffer", lifetime="engine", scope="group_ipc"
            ),
        ),
    ],
)
def test_one_destroy_export_cannot_mix_resource_lifetimes_or_scopes(resources):
    plan = ArtifactResourcePlan(
        slot=COLLECTIVE_ALL_REDUCE_CALL_ABI.slot,
        resources=resources,
    )
    destroy = tuple(ArtifactBinding(resource.name, "tensor") for resource in resources)

    with pytest.raises(ArtifactABIError, match="different lifetime/scope"):
        plan.validate_pipeline((("destroy", destroy, ()),))


def test_one_destroy_export_may_cover_one_lifetime_scope_class():
    resources = (
        _static_resource("state.first", lifetime="engine"),
        _static_resource("state.second", lifetime="engine"),
    )
    plan = ArtifactResourcePlan(
        slot=COLLECTIVE_ALL_REDUCE_CALL_ABI.slot,
        resources=resources,
    )
    destroy = tuple(ArtifactBinding(resource.name, "tensor") for resource in resources)

    plan.validate_pipeline((("destroy", destroy, ()),))


def test_peer_ptr_table_seals_exact_persistent_group_ipc_buffer():
    plan = ArtifactResourcePlan(
        slot=COLLECTIVE_ALL_REDUCE_CALL_ABI.slot,
        resources=(
            _static_resource(
                "state.collective_buffer",
                lifetime="engine",
                extent=4096,
                scope="group_ipc",
            ),
        ),
    )
    bindings = (
        ArtifactBinding("output.out", "tensor"),
        ArtifactBinding(
            "group.current",
            "pointer",
            projection="peer_ptr_table",
            peer_resource="state.collective_buffer",
        ),
    )

    COLLECTIVE_ALL_REDUCE_CALL_ABI.validate_plan(
        role="run",
        bindings=bindings,
        specializes={},
        prelaunch=(),
        artifact_resources=plan,
    )
    plan.validate_pipeline((("run", bindings, ()),), require_all=True)

    parsed = parse_artifact_bindings(
        (bindings[1].to_dict(),), field="artifact.bindings"
    )
    assert parsed == (bindings[1],)
    assert parsed[0].peer_resource == "state.collective_buffer"


def test_peer_ptr_table_rejects_missing_undeclared_or_rank_local_buffer():
    with pytest.raises(ArtifactABIError, match="peer_resource"):
        ArtifactBinding(
            "group.current", "pointer", projection="peer_ptr_table"
        )
    with pytest.raises(ArtifactABIError, match="canonical slot resource"):
        ArtifactBinding(
            "group.current",
            "pointer",
            projection="peer_ptr_table",
            peer_resource="candidate.exchange()",
        )

    rank_local = ArtifactResourcePlan(
        slot=COLLECTIVE_ALL_REDUCE_CALL_ABI.slot,
        resources=(
            _static_resource("state.collective_buffer", lifetime="engine"),
        ),
    )
    for peer_resource, message in (
        ("state.not_declared", "undeclared artifact buffer"),
        ("state.collective_buffer", "persistent group-IPC"),
    ):
        with pytest.raises(ArtifactABIError, match=message):
            COLLECTIVE_ALL_REDUCE_CALL_ABI.validate_plan(
                role="run",
                bindings=(
                    ArtifactBinding("output.out", "tensor"),
                    ArtifactBinding(
                        "group.current",
                        "pointer",
                        projection="peer_ptr_table",
                        peer_resource=peer_resource,
                    ),
                ),
                specializes={},
                prelaunch=(),
                artifact_resources=rank_local,
            )


def test_peer_ptr_table_requires_group_capability_and_native_handle_stays_group_only():
    abi = SlotCallABI(
        slot="collective.no_peer_capability",
        resources=(
            SlotResource("input.x", "tensor"),
            SlotResource("output.out", "tensor", access="write"),
            SlotResource("group.current", "group"),
        ),
        call_args=("input.x", "output.out", "group.current"),
    )
    plan = ArtifactResourcePlan(
        slot=abi.slot,
        resources=(
            _static_resource(
                "state.collective_buffer",
                lifetime="engine",
                scope="group_ipc",
            ),
        ),
    )
    with pytest.raises(ArtifactABIError, match="requires validator capability"):
        abi.validate_plan(
            role="run",
            bindings=(
                ArtifactBinding("output.out", "tensor"),
                ArtifactBinding(
                    "group.current",
                    "pointer",
                    projection="peer_ptr_table",
                    peer_resource="state.collective_buffer",
                ),
            ),
            specializes={},
            prelaunch=(),
            artifact_resources=plan,
        )

    with pytest.raises(ArtifactABIError, match="not allowed.*kind 'tensor'"):
        COLLECTIVE_ALL_REDUCE_CALL_ABI.validate_plan(
            role="run",
            bindings=(
                ArtifactBinding("output.out", "tensor"),
                ArtifactBinding(
                    "input.x", "pointer", projection="native_handle"
                ),
            ),
            specializes={},
            prelaunch=(),
        )
    with pytest.raises(ArtifactABIError, match="only for a peer_ptr_table"):
        ArtifactBinding(
            "group.current",
            "pointer",
            projection="native_handle",
            peer_resource="state.collective_buffer",
        )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda row: {**row, "name": "input.miner_owned"},
            "workspace.*",
        ),
        (
            lambda row: {**row, "lifetime": "engine"},
            "requires lifetime 'call'",
        ),
        (
            lambda row: {**row, "alignment": 3},
            "power of two",
        ),
        (
            lambda row: {**row, "python_callback": "miner.allocate"},
            "unknown=.*python_callback",
        ),
    ],
)
def test_resource_schema_rejects_open_ended_authority(mutate, message):
    row = _resource_rows()[0]
    with pytest.raises(ArtifactABIError, match=message):
        parse_artifact_resources(
            (mutate(row),),
            call_abi=MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
            field="artifact_resources",
        )


def test_resource_shapes_cannot_depend_on_generated_or_unknown_values():
    row = _resource_rows()[0]
    shape = (
        {
            "factors": (
                {
                    "source": "workspace.other",
                    "projection": "numel",
                    "upper_bound": 1024,
                },
            ),
        },
    )
    with pytest.raises(ArtifactABIError, match="unknown validator slot resource"):
        parse_artifact_resources(
            ({**row, "shape": shape},),
            call_abi=MSA_PREFILL_BLOCK_SCORE_CALL_ABI,
            field="artifact_resources",
        )


def test_resource_namespace_cannot_shadow_semantic_prepared_state():
    row = {
        **_resource_rows()[2],
        "name": "prepared.state",
        "shape": ({"factors": ({"static": 1},)},),
    }
    with pytest.raises(ArtifactABIError, match="collide"):
        parse_artifact_resources(
            (row,),
            call_abi=MOE_FUSED_EXPERTS_CALL_ABI,
            field="artifact_resources",
        )


def test_moe_prepare_boundary_can_return_a_validator_owned_artifact_frame():
    plan = parse_artifact_resources(
        (
            {
                "name": "prepared.packed_weights",
                "dtype": "u8",
                "alignment": 128,
                "lifetime": "prepared",
                "shape": (
                    {
                        "factors": (
                            {
                                "source": "input.w13",
                                "projection": "numel",
                                "upper_bound": 1 << 30,
                            },
                        ),
                        "divisor": 2,
                    },
                ),
            },
        ),
        call_abi=MOE_FUSED_EXPERTS_CALL_ABI,
        field="artifact_resources",
    )
    prepare = (
        ArtifactBinding("input.w13", "tensor"),
        ArtifactBinding("input.w2", "tensor"),
        ArtifactBinding("prepared.packed_weights", "tensor"),
    )
    run = (
        ArtifactBinding("prepared.packed_weights", "tensor"),
        ArtifactBinding("output.out", "tensor"),
    )
    MOE_FUSED_EXPERTS_CALL_ABI.validate_plan(
        role="prepare",
        bindings=prepare,
        specializes={},
        prelaunch=(),
        require_outputs=False,
        artifact_resources=plan,
    )
    MOE_FUSED_EXPERTS_CALL_ABI.validate_plan(
        role="run",
        bindings=run,
        specializes={},
        prelaunch=(),
        artifact_resources=plan,
    )
    MOE_FUSED_EXPERTS_CALL_ABI.validate_pipeline(
        (("prepare", prepare, ()), ("run", run, ())),
        artifact_resources=plan,
    )


def test_native_run_cannot_bind_opaque_semantic_prepare_envelope():
    with pytest.raises(ArtifactABIError, match="prepare envelope directly"):
        MOE_FUSED_EXPERTS_CALL_ABI.validate_plan(
            role="run",
            bindings=(
                ArtifactBinding("prepared.state", "opaque"),
                ArtifactBinding("output.out", "tensor"),
            ),
            specializes={},
            prelaunch=(),
        )


def test_resource_count_per_buffer_and_total_bytes_are_hard_bounded():
    tiny = tuple(
        _static_resource(f"workspace.r{index}", lifetime="call")
        for index in range(33)
    )
    with pytest.raises(ArtifactABIError, match="at most 32"):
        ArtifactResourcePlan(
            slot=MSA_PREFILL_BLOCK_SCORE_CALL_ABI.slot,
            resources=tiny,
        )

    with pytest.raises(ArtifactABIError, match="per-buffer byte cap"):
        ArtifactResource(
            name="state.too_large",
            dtype="f64",
            alignment=16,
            lifetime="engine",
            shape=(
                ArtifactShapeExtent(
                    factors=(ArtifactShapeFactor(static=1 << 18),),
                ),
                ArtifactShapeExtent(
                    factors=(ArtifactShapeFactor(static=1 << 18),),
                ),
            ),
        )

    sixty_four_gib = ArtifactResource(
        name="state.a",
        dtype="u8",
        alignment=16,
        lifetime="engine",
        shape=(
            ArtifactShapeExtent(
                factors=(ArtifactShapeFactor(static=1 << 18),),
            ),
            ArtifactShapeExtent(
                factors=(ArtifactShapeFactor(static=1 << 18),),
            ),
        ),
    )
    with pytest.raises(ArtifactABIError, match="aggregate byte cap"):
        ArtifactResourcePlan(
            slot=MSA_PREFILL_BLOCK_SCORE_CALL_ABI.slot,
            resources=(
                sixty_four_gib,
                _static_resource("state.b", lifetime="engine"),
                ArtifactResource(
                    name="state.c",
                    dtype="u8",
                    alignment=16,
                    lifetime="engine",
                    shape=sixty_four_gib.shape,
                ),
            ),
        )
