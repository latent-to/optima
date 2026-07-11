"""Unit coverage for the validator-owned typed tensor allocation contract."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from optima.slots import get_slot  # noqa: E402
from optima.tensor_spec import (  # noqa: E402
    OutputSpec,
    TensorSpec,
    allocate_output_spec,
    validate_allocation_bindings,
    validate_output_spec,
    validate_tensor,
)


def test_legacy_slot_contract_stays_contiguous_and_inherits_dtype_device():
    slot = get_slot("activation.silu_and_mul")
    inputs = slot.make_inputs(
        num_tokens=2, d=8, dtype=torch.float32, device="cpu", seed=0
    )
    contract = slot.output_contract(inputs)
    assert contract.outputs[0].dtype is None
    assert contract.outputs[0].device is None
    assert contract.outputs[0].stride_policy == "contiguous"

    allocation = allocate_output_spec(
        contract,
        fallback_dtype=torch.float32,
        fallback_device="cpu",
        inputs=inputs.values(),
    )
    assert allocation.outputs[0].dtype == torch.float32
    assert allocation.outputs[0].is_contiguous()


def test_strided_output_and_workspace_are_validator_allocated_and_aligned():
    contract = OutputSpec(
        outputs=(TensorSpec(
            shape=(3, 5),
            dtype=torch.float32,
            stride_policy="strided",
            stride_padding=3,
            alignment_bytes=64,
            name="scores",
        ),),
        workspace=(TensorSpec(
            shape=(17,), dtype=torch.uint8, alignment_bytes=64, name="scratch"
        ),),
    )
    allocation = allocate_output_spec(
        contract,
        fallback_dtype=torch.bfloat16,
        fallback_device="cpu",
    )
    out = allocation.outputs[0]
    scratch = allocation.workspace[0]
    assert out.dtype == torch.float32
    assert out.stride() == (8, 1)
    assert not out.is_contiguous()
    assert out.data_ptr() % 64 == 0
    assert scratch.dtype == torch.uint8
    assert scratch.data_ptr() % 64 == 0


def test_disjoint_alias_policy_rejects_shared_storage():
    source = torch.empty(8, dtype=torch.float32)
    view = source[:4]
    spec = TensorSpec(shape=(4,), dtype=torch.float32, aliasing="disjoint")
    with pytest.raises(ValueError, match="aliases"):
        validate_tensor(
            view,
            spec,
            fallback_dtype=torch.float32,
            fallback_device="cpu",
            disjoint_from=(source,),
        )


def test_disjoint_sibling_policy_is_symmetric():
    storage = torch.empty(8, dtype=torch.float32)
    contract = OutputSpec(
        outputs=(
            TensorSpec(
                shape=(4,), dtype=torch.float32, aliasing="disjoint", name="first"
            ),
            TensorSpec(
                shape=(4,), dtype=torch.float32, aliasing="may_alias", name="second"
            ),
        )
    )
    with pytest.raises(ValueError, match="aliases"):
        validate_output_spec(
            contract,
            (storage[:4], storage[4:]),
            fallback_dtype=torch.float32,
            fallback_device="cpu",
        )


def test_strided_row_major_policy_rejects_holes_between_columns():
    storage = torch.empty(64, dtype=torch.float32)
    column_gapped = storage.as_strided((3, 5), (10, 2))
    spec = TensorSpec(
        shape=(3, 5), dtype=torch.float32, stride_policy="strided"
    )

    with pytest.raises(ValueError, match="row-major strided"):
        validate_tensor(
            column_gapped,
            spec,
            fallback_dtype=torch.float32,
            fallback_device="cpu",
        )


def test_validator_owned_storage_binding_rejects_set_replacement():
    contract = OutputSpec((TensorSpec(shape=(3, 5), dtype=torch.float32),))
    allocation = allocate_output_spec(
        contract, fallback_dtype=torch.float32, fallback_device="cpu"
    )
    out = allocation.outputs[0]
    replacement = torch.empty_like(out)
    out.set_(replacement)

    # Shape/dtype/stride alone cannot distinguish the replacement; the retained
    # original storage identity can.
    validate_output_spec(
        contract,
        allocation.outputs,
        fallback_dtype=torch.float32,
        fallback_device="cpu",
    )
    with pytest.raises(ValueError, match="validator-owned storage"):
        validate_allocation_bindings(allocation)


def test_tensor_binding_rejects_in_place_stride_change():
    contract = OutputSpec((TensorSpec(shape=(3, 5), dtype=torch.float32),))
    allocation = allocate_output_spec(
        contract, fallback_dtype=torch.float32, fallback_device="cpu"
    )
    allocation.outputs[0].as_strided_((3, 5), (1, 3))

    with pytest.raises(ValueError, match="validator-owned storage/tensor binding"):
        validate_allocation_bindings(allocation)
