"""Validator-owned tensor allocation contracts.

``SlotSpec.out_shapes`` predates typed outputs and remains the compatibility ABI.
This module is the additive foundation for slots whose live output is more specific
than "an input-dtype contiguous tensor": dtype, device, layout, stride tolerance,
alignment, and storage aliasing are declared once and checked by both verification
and the live arena binding.

The contract deliberately implements only policies the validator can enforce.  In
particular, ``aliasing`` is either ``"disjoint"`` (the default, checked against
inputs and sibling buffers) or ``"may_alias"``.  ``OutputSpec.workspace`` is
validator-allocated and kept alive with the outputs, but is not silently passed to
legacy entry callables; a future slot/region ABI must explicitly consume it.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Any, Iterable, Optional, Sequence

import torch


_STRIDE_POLICIES = frozenset(("contiguous", "strided"))
_ALIAS_POLICIES = frozenset(("disjoint", "may_alias"))


@dataclass(frozen=True)
class TensorSpec:
    """A resolved logical tensor contract.

    ``dtype`` and ``device`` may be ``None`` to inherit the verifier/arena call's
    defaults.  ``stride_policy="strided"`` means a row-major, non-overlapping
    strided view is legal; verification intentionally allocates it with padded row
    strides so kernels cannot accidentally assume contiguity.  ``stride_padding``
    controls only that adversarial verification allocation, not the set of legal
    live row strides.
    """

    shape: tuple[int, ...]
    dtype: Optional[torch.dtype] = None
    device: Optional[str | torch.device] = None
    layout: torch.layout = torch.strided
    stride_policy: str = "contiguous"
    stride_padding: int = 0
    alignment_bytes: int = 1
    aliasing: str = "disjoint"
    name: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))
        if any(d < 0 for d in self.shape):
            raise ValueError(f"negative tensor dimension in {self.shape}")
        if self.layout is not torch.strided:
            raise ValueError("only torch.strided tensor contracts are currently enforceable")
        if self.stride_policy not in _STRIDE_POLICIES:
            raise ValueError(
                f"unknown stride_policy {self.stride_policy!r}; expected one of "
                f"{sorted(_STRIDE_POLICIES)}"
            )
        if self.stride_padding < 0:
            raise ValueError("stride_padding must be non-negative")
        if self.stride_policy == "contiguous" and self.stride_padding:
            raise ValueError("stride_padding is only valid for stride_policy='strided'")
        if self.alignment_bytes < 1 or self.alignment_bytes & (self.alignment_bytes - 1):
            raise ValueError("alignment_bytes must be a positive power of two")
        if self.aliasing not in _ALIAS_POLICIES:
            raise ValueError(
                f"unknown aliasing policy {self.aliasing!r}; expected one of "
                f"{sorted(_ALIAS_POLICIES)}"
            )


@dataclass(frozen=True)
class OutputSpec:
    """All validator-owned buffers for one slot invocation."""

    outputs: tuple[TensorSpec, ...]
    workspace: tuple[TensorSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "workspace", tuple(self.workspace))
        if not self.outputs:
            raise ValueError("an OutputSpec must declare at least one output")


@dataclass
class TensorAllocation:
    """Owns output/workspace views and their backing storages."""

    outputs: list[torch.Tensor]
    workspace: list[torch.Tensor]
    output_bindings: tuple["TensorStorageBinding", ...] = ()
    workspace_bindings: tuple["TensorStorageBinding", ...] = ()


@dataclass(frozen=True)
class TensorStorageBinding:
    """Exact validator-owned storage identity captured before candidate code runs.

    Keeping the original storage object alive prevents allocator reuse from making a
    miner-installed replacement appear to have the same address. Shape/stride/type
    remain the responsibility of :func:`validate_output_spec`; this binds the tensor
    object to the allocation the validator actually created.
    """

    storage: Any
    storage_cdata: int
    storage_ptr: int
    storage_nbytes: int
    data_ptr: int
    storage_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout


def tensor_storage_binding(tensor: torch.Tensor) -> TensorStorageBinding:
    storage = tensor.untyped_storage()
    return TensorStorageBinding(
        storage=storage,
        storage_cdata=int(storage._cdata),
        storage_ptr=int(storage.data_ptr()),
        storage_nbytes=int(storage.nbytes()),
        data_ptr=int(tensor.data_ptr()),
        storage_offset=int(tensor.storage_offset()),
        shape=tuple(tensor.shape),
        stride=tuple(tensor.stride()),
        dtype=tensor.dtype,
        device=tensor.device,
        layout=tensor.layout,
    )


def validate_tensor_binding(
    tensor: torch.Tensor,
    expected: TensorStorageBinding,
    *,
    name: str = "tensor",
) -> None:
    """Require the exact pre-candidate storage and tensor metadata."""

    storage = tensor.untyped_storage()
    actual = (
        int(storage._cdata),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.data_ptr()),
        int(tensor.storage_offset()),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
        tensor.layout,
    )
    wanted = (
        expected.storage_cdata,
        expected.storage_ptr,
        expected.storage_nbytes,
        expected.data_ptr,
        expected.storage_offset,
        expected.shape,
        expected.stride,
        expected.dtype,
        expected.device,
        expected.layout,
    )
    if actual != wanted:
        raise ValueError(
            f"{name} no longer uses its validator-owned storage/tensor binding"
        )


def validate_allocation_bindings(allocation: TensorAllocation) -> None:
    """Reject ``set_``/storage replacement of validator-owned live buffers."""

    groups = (
        ("output", allocation.outputs, allocation.output_bindings),
        ("workspace", allocation.workspace, allocation.workspace_bindings),
    )
    for kind, tensors, bindings in groups:
        if len(tensors) != len(bindings):
            raise ValueError(
                f"{kind} storage binding count {len(bindings)} != tensor count "
                f"{len(tensors)}"
            )
        for index, (tensor, expected) in enumerate(zip(tensors, bindings)):
            validate_tensor_binding(tensor, expected, name=f"{kind}[{index}]")


def tensor_bindings(
    tensors: Iterable[torch.Tensor],
) -> tuple[TensorStorageBinding, ...]:
    """Capture exact bindings while retaining every original storage strongly."""

    return tuple(tensor_storage_binding(tensor) for tensor in tensors)


def validate_tensor_bindings(
    tensors: Iterable[torch.Tensor],
    bindings: Sequence[TensorStorageBinding],
    *,
    kind: str = "input",
) -> None:
    tensors = tuple(tensors)
    if len(tensors) != len(bindings):
        raise ValueError(
            f"{kind} binding count {len(bindings)} != tensor count {len(tensors)}"
        )
    for index, (tensor, binding) in enumerate(zip(tensors, bindings)):
        validate_tensor_binding(tensor, binding, name=f"{kind}[{index}]")


def validate_output_allocation(
    spec: OutputSpec,
    allocation: TensorAllocation,
    *,
    fallback_dtype: torch.dtype,
    fallback_device: str | torch.device,
    inputs: Iterable[torch.Tensor] = (),
) -> None:
    """Revalidate storage and the complete typed ABI after candidate execution."""

    validate_allocation_bindings(allocation)
    inputs = tuple(tensor for tensor in inputs if torch.is_tensor(tensor))
    validate_output_spec(
        spec,
        allocation.outputs,
        fallback_dtype=fallback_dtype,
        fallback_device=fallback_device,
        inputs=inputs,
    )
    if len(allocation.workspace) != len(spec.workspace):
        raise ValueError(
            f"workspace count {len(allocation.workspace)} != declared "
            f"{len(spec.workspace)}"
        )
    checked: list[torch.Tensor] = list(allocation.outputs)
    for tensor, tensor_spec in zip(allocation.workspace, spec.workspace):
        validate_tensor(
            tensor,
            tensor_spec,
            fallback_dtype=fallback_dtype,
            fallback_device=fallback_device,
            disjoint_from=(*inputs, *checked),
        )
        checked.append(tensor)


def _resolved_dtype(spec: TensorSpec, fallback: torch.dtype) -> torch.dtype:
    return spec.dtype if spec.dtype is not None else fallback


def _resolved_device(spec: TensorSpec, fallback: str | torch.device) -> torch.device:
    return torch.device(spec.device if spec.device is not None else fallback)


def _representative_strides(spec: TensorSpec) -> tuple[int, ...]:
    shape = spec.shape
    if not shape:
        return ()
    if spec.stride_policy == "contiguous":
        stride = 1
        out = [0] * len(shape)
        for i in range(len(shape) - 1, -1, -1):
            out[i] = stride
            stride *= max(shape[i], 1)
        return tuple(out)

    # A padded row-major view.  Always add at least one element of padding so a
    # strided-capable contract is actually exercised as non-contiguous in verify.
    padding = max(1, spec.stride_padding)
    if len(shape) == 1:
        return (1 + padding,)
    out = [0] * len(shape)
    out[-1] = 1
    out[-2] = max(shape[-1], 1) + padding
    for i in range(len(shape) - 3, -1, -1):
        out[i] = out[i + 1] * max(shape[i + 1], 1)
    return tuple(out)


def _storage_elements(shape: Sequence[int], strides: Sequence[int]) -> int:
    if any(d == 0 for d in shape):
        return 0
    if not shape:
        return 1
    return 1 + sum((d - 1) * s for d, s in zip(shape, strides))


def _aligned_offset(base: torch.Tensor, alignment_bytes: int) -> int:
    if alignment_bytes == 1:
        return 0
    itemsize = base.element_size()
    attempts = alignment_bytes // gcd(alignment_bytes, itemsize)
    ptr = base.data_ptr()
    for offset in range(attempts):
        if (ptr + offset * itemsize) % alignment_bytes == 0:
            return offset
    raise RuntimeError(
        f"cannot align {itemsize}-byte elements to {alignment_bytes} bytes"
    )


def allocate_tensor(
    spec: TensorSpec,
    *,
    fallback_dtype: torch.dtype,
    fallback_device: str | torch.device,
) -> torch.Tensor:
    """Allocate the validator's representative tensor for ``spec``."""

    dtype = _resolved_dtype(spec, fallback_dtype)
    device = _resolved_device(spec, fallback_device)
    strides = _representative_strides(spec)
    storage_elements = _storage_elements(spec.shape, strides)
    # Element size is a dtype property; do not create a throwaway device allocation
    # (especially not a CUDA allocation) merely to query it.
    itemsize = torch.empty((), dtype=dtype).element_size()
    alignment_slack = max(1, spec.alignment_bytes // gcd(spec.alignment_bytes, itemsize))
    base = torch.empty(max(1, storage_elements) + alignment_slack, dtype=dtype, device=device)
    offset = _aligned_offset(base, spec.alignment_bytes)
    out = base.as_strided(spec.shape, strides, storage_offset=offset)
    validate_tensor(
        out,
        spec,
        fallback_dtype=fallback_dtype,
        fallback_device=fallback_device,
    )
    return out


def _same_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    if a.numel() == 0 or b.numel() == 0:
        return False
    try:
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
    except (AttributeError, RuntimeError):
        return False


def _device_matches(actual: torch.device, expected: torch.device) -> bool:
    if actual.type != expected.type:
        return False
    return expected.index is None or actual.index == expected.index


def _row_major_non_overlapping(tensor: torch.Tensor) -> bool:
    """Whether ``tensor`` has a non-overlapping row-major positive-stride view."""

    if tensor.layout is not torch.strided:
        return False
    if tensor.dim() == 0:
        return True
    shape = tensor.shape
    strides = tensor.stride()
    if any(s <= 0 for d, s in zip(shape, strides) if d > 1):
        return False
    # "strided" means dense columns with an optional padded row pitch, not an
    # arbitrary non-overlapping permutation or holes between adjacent columns.
    if tensor.dim() >= 2 and shape[-1] > 1 and strides[-1] != 1:
        return False
    inner_span = 1
    for dim, stride in reversed(list(zip(shape, strides))):
        if dim <= 1:
            continue
        if stride < inner_span:
            return False
        inner_span += (dim - 1) * stride
    return True


def validate_tensor(
    tensor: torch.Tensor,
    spec: TensorSpec,
    *,
    fallback_dtype: torch.dtype,
    fallback_device: str | torch.device,
    disjoint_from: Iterable[torch.Tensor] = (),
) -> None:
    """Fail closed when a live/verify tensor violates its declared contract."""

    if tuple(tensor.shape) != spec.shape:
        raise ValueError(
            f"{spec.name or 'tensor'} shape {tuple(tensor.shape)} != declared {spec.shape}"
        )
    dtype = _resolved_dtype(spec, fallback_dtype)
    if tensor.dtype != dtype:
        raise ValueError(
            f"{spec.name or 'tensor'} dtype {tensor.dtype} != declared {dtype}"
        )
    device = _resolved_device(spec, fallback_device)
    if not _device_matches(tensor.device, device):
        raise ValueError(
            f"{spec.name or 'tensor'} device {tensor.device} != declared {device}"
        )
    if tensor.layout is not spec.layout:
        raise ValueError(
            f"{spec.name or 'tensor'} layout {tensor.layout} != declared {spec.layout}"
        )
    if spec.stride_policy == "contiguous":
        if not tensor.is_contiguous():
            raise ValueError(f"{spec.name or 'tensor'} must be contiguous")
    elif not _row_major_non_overlapping(tensor):
        raise ValueError(
            f"{spec.name or 'tensor'} must be a non-overlapping row-major strided view; "
            f"got stride={tensor.stride()}"
        )
    if tensor.data_ptr() % spec.alignment_bytes:
        raise ValueError(
            f"{spec.name or 'tensor'} pointer is not {spec.alignment_bytes}-byte aligned"
        )
    if spec.aliasing == "disjoint":
        for other in disjoint_from:
            if torch.is_tensor(other) and _same_storage(tensor, other):
                raise ValueError(f"{spec.name or 'tensor'} aliases another contract tensor")


def allocate_output_spec(
    spec: OutputSpec,
    *,
    fallback_dtype: torch.dtype,
    fallback_device: str | torch.device,
    inputs: Iterable[torch.Tensor] = (),
) -> TensorAllocation:
    """Allocate and cross-check every declared output and workspace buffer."""

    inputs = tuple(t for t in inputs if torch.is_tensor(t))
    allocated: list[torch.Tensor] = []
    outputs: list[torch.Tensor] = []
    workspace: list[torch.Tensor] = []
    for tensor_spec, destination in (
        *((s, outputs) for s in spec.outputs),
        *((s, workspace) for s in spec.workspace),
    ):
        tensor = allocate_tensor(
            tensor_spec,
            fallback_dtype=fallback_dtype,
            fallback_device=fallback_device,
        )
        validate_tensor(
            tensor,
            tensor_spec,
            fallback_dtype=fallback_dtype,
            fallback_device=fallback_device,
            disjoint_from=(*inputs, *allocated),
        )
        destination.append(tensor)
        allocated.append(tensor)
    return TensorAllocation(
        outputs=outputs,
        workspace=workspace,
        output_bindings=tuple(tensor_storage_binding(tensor) for tensor in outputs),
        workspace_bindings=tuple(
            tensor_storage_binding(tensor) for tensor in workspace
        ),
    )


def validate_output_spec(
    spec: OutputSpec,
    outputs: Sequence[torch.Tensor],
    *,
    fallback_dtype: torch.dtype,
    fallback_device: str | torch.device,
    inputs: Iterable[torch.Tensor] = (),
) -> None:
    """Validate live arena output views against the same declaration as verify."""

    if len(outputs) != len(spec.outputs):
        raise ValueError(
            f"output count {len(outputs)} != declared {len(spec.outputs)}"
        )
    inputs = tuple(t for t in inputs if torch.is_tensor(t))
    peers: list[tuple[torch.Tensor, TensorSpec]] = []
    for tensor, tensor_spec in zip(outputs, spec.outputs):
        validate_tensor(
            tensor,
            tensor_spec,
            fallback_dtype=fallback_dtype,
            fallback_device=fallback_device,
            disjoint_from=inputs,
        )
        for peer, peer_spec in peers:
            if (
                tensor_spec.aliasing == "disjoint"
                or peer_spec.aliasing == "disjoint"
            ) and _same_storage(tensor, peer):
                raise ValueError(
                    f"{tensor_spec.name or 'tensor'} aliases another contract tensor"
                )
        peers.append((tensor, tensor_spec))
