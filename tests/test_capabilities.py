"""Declarative specialization domains and validator-owned routing."""

from __future__ import annotations

import pytest

from optima.capabilities import (
    CallDescriptor,
    CapabilityMetadataError,
    capability_domain_from_metadata,
)
from optima.registry import (
    Eligibility,
    KernelImpl,
    KernelRegistry,
    SelectionOutcome,
    VariantRegistrationError,
    eligibility_from_metadata,
)


def test_call_descriptor_canonicalizes_aliases_and_values():
    desc = CallDescriptor.from_mapping(
        {
            "dtype_name": "torch.bfloat16",
            "arch": "SM_103",
            "tp": 4,
            "phase": "extend",
            "graph-mode": "capture",
            "quant": "unquantized",
            "model": "MiniMax-M3-NVFP4",
        }
    )
    assert desc.as_dict() == {
        "architecture": "sm103",
        "dtype": "bfloat16",
        "graph_mode": "cuda_graph",
        "model": "MiniMax-M3-NVFP4",
        "phase": "prefill",
        "quant": "dense",
        "tp_size": 4,
    }
    assert desc["arch"] == "sm103"


def test_call_descriptor_rejects_conflicting_aliases():
    with pytest.raises(ValueError, match="conflicting values.*tp_size"):
        CallDescriptor({"tp": 4, "tp_size": 8})


def test_domain_exact_enumerated_and_range_match():
    domain = capability_domain_from_metadata(
        {
            "capabilities": {
                "head_dim": 128,  # scalar shorthand -> exact
                "block_size": {"one_of": [64, 128]},
                "q_len": {"min": 1, "max": 32768},
                "phase": ["prefill"],  # list shorthand -> one_of
                "layout": {"exact": "paged-kv"},
                "tp": 4,
                "ep_size": {"min": 1, "max": 8},
                "world_size": {"one_of": [4, 8]},
                "model": "MiniMax-M3-NVFP4",
                "dtype": ["bf16"],
                "quant": "none",
                "arch": "sm_103",
                "graph_mode": ["eager", "capture"],
            }
        }
    )
    desc = CallDescriptor(
        head_dim=128,
        block_size=128,
        q_len=4096,
        phase="extend",
        layout="paged_kv",
        tp_size=4,
        ep=2,
        world=8,
        model="MiniMax-M3-NVFP4",
        dtype="bfloat16",
        quant="dense",
        architecture="SM103",
        graph_mode="cuda_graph",
    )
    assert domain.match(desc).accepted


def test_domain_missing_fields_and_mismatches_fail_closed():
    domain = capability_domain_from_metadata(
        {"capabilities": {"head_dim": 128, "phase": ["prefill"]}}
    )
    match = domain.match(CallDescriptor(head_dim=64))
    assert not match.accepted
    assert [(m.field, m.reason, m.actual) for m in match.mismatches] == [
        ("head_dim", "outside_domain", 64),
        ("phase", "missing", None),
    ]


@pytest.mark.parametrize(
    "capabilities, message",
    [
        ({"hed_dim": 128}, "unsupported capability field"),
        ({"head_dim": {"min": 129, "max": 128}}, "greater than"),
        ({"phase": {"min": 1}}, "contextual and cannot use min/max"),
        ({"head_dim": {"exact": 128, "max": 256}}, "cannot mix"),
        ({"head_dim": {"one_of": []}}, "non-empty list"),
        ({"head_dim": True}, "must be an integer"),
    ],
)
def test_invalid_capability_metadata_rejected(capabilities, message):
    with pytest.raises(CapabilityMetadataError, match=message):
        capability_domain_from_metadata({"capabilities": capabilities})


@pytest.mark.parametrize("metadata", [[], "", 0, False])
def test_falsy_nonobject_capability_metadata_fails_closed(metadata):
    with pytest.raises(CapabilityMetadataError, match="JSON object"):
        capability_domain_from_metadata(metadata)
    with pytest.raises(ValueError, match="JSON object"):
        eligibility_from_metadata(metadata, ())


def test_legacy_eligibility_metadata_is_preserved():
    eligibility = eligibility_from_metadata(
        {
            "dtypes": ["bfloat16"],
            "architectures": ["sm103"],
            "max_last_dim": 128,
            "min_num_tokens": 48,
            "max_num_tokens": 1024,
            # Descriptive prose is not silently promoted to a dispatch gate.
            "regime": {"head_dim": 128, "phase": "prefill"},
        },
        ("bfloat16",),
    )
    assert eligibility.capabilities.predicates == ()
    assert eligibility.accepts(
        dtype_name="bfloat16", last_dim=128, arch="sm103", num_tokens=48
    )
    # Legacy behavior: unknown arch/token count was not a rejection.
    assert eligibility.accepts(
        dtype_name="bfloat16", last_dim=128, arch=None, num_tokens=None
    )
    assert not eligibility.accepts(
        dtype_name="bfloat16", last_dim=129, arch="sm103", num_tokens=48
    )


def test_new_capability_metadata_requires_binding_to_supply_fields():
    eligibility = eligibility_from_metadata(
        {"capabilities": {"head_dim": 128, "block_size": 128}},
        ("bfloat16",),
    )
    # An old dispatcher does not know these fields and therefore serves stock.
    assert not eligibility.accepts(
        dtype_name="bfloat16", last_dim=128, arch="sm103"
    )
    assert eligibility.match(
        CallDescriptor(
            dtype="bfloat16", head_dim=128, block_size=128, last_dim=128
        )
    ).accepted


def _registry(eligibility: Eligibility) -> KernelRegistry:
    registry = KernelRegistry()
    registry.register(
        KernelImpl(
            slot="attention.msa_prefill_block_score",
            bundle_id="candidate",
            entry=lambda *_args: None,
            eligibility=eligibility,
        )
    )
    return registry


def test_registry_selection_exposes_validator_owned_fallback_reasons():
    eligibility = eligibility_from_metadata(
        {
            "graph_safe": False,
            "quant": ["nvfp4"],
            "capabilities": {
                "head_dim": 128,
                "block_size": 128,
                "phase": "prefill",
                "tp_size": 4,
            },
        },
        ("bfloat16",),
    )
    registry = _registry(eligibility)
    good = CallDescriptor(
        dtype="bfloat16",
        architecture="sm103",
        quant="nvfp4",
        graph_mode="eager",
        head_dim=128,
        block_size=128,
        phase="prefill",
        tp_size=4,
    )

    inactive = registry.select("attention.msa_prefill_block_score", good)
    assert inactive.outcome is SelectionOutcome.REGISTRY_INACTIVE
    assert inactive.use_baseline

    registry.enable()
    missing = registry.select("unknown.slot", good)
    assert missing.outcome is SelectionOutcome.SLOT_UNREGISTERED
    assert missing.use_baseline

    wrong = registry.select(
        "attention.msa_prefill_block_score",
        good.with_updates(head_dim=64),
        write_fired_receipt=False,
    )
    assert wrong.outcome is SelectionOutcome.OUT_OF_DOMAIN
    assert wrong.candidate is not None and wrong.impl is None
    assert [m.field for m in wrong.capability_match.mismatches] == ["head_dim"]

    selected = registry.select(
        "attention.msa_prefill_block_score", good, write_fired_receipt=False
    )
    assert selected.outcome is SelectionOutcome.SELECTED
    assert selected.use_candidate and selected.impl is selected.candidate


def test_graph_and_quant_context_are_enforced_by_canonical_selection():
    registry = _registry(
        Eligibility(
            dtypes=frozenset({"bfloat16"}),
            quant=frozenset({"nvfp4"}),
            graph_safe=False,
        )
    )
    registry.enable()
    base = CallDescriptor(dtype="bfloat16", quant="nvfp4", graph_mode="eager")
    assert registry.select(
        "attention.msa_prefill_block_score", base, write_fired_receipt=False
    ).use_candidate

    graph = registry.select(
        "attention.msa_prefill_block_score",
        base.with_updates(graph_mode="cuda_graph"),
        write_fired_receipt=False,
    )
    assert graph.outcome is SelectionOutcome.OUT_OF_DOMAIN
    assert [m.field for m in graph.capability_match.mismatches] == ["graph_mode"]

    quant = registry.select(
        "attention.msa_prefill_block_score",
        base.with_updates(quant="dense"),
        write_fired_receipt=False,
    )
    assert quant.outcome is SelectionOutcome.OUT_OF_DOMAIN
    assert [m.field for m in quant.capability_match.mismatches] == ["quant"]


def test_legacy_domain_spellings_use_descriptor_canonicalization():
    eligibility = eligibility_from_metadata(
        {"architectures": ["sm_103"]},
        ("bf16",),
    )
    assert eligibility.match(
        CallDescriptor(dtype="bfloat16", architecture="sm103")
    ).accepted
    assert eligibility.accepts(
        dtype_name="torch.bfloat16", last_dim=128, arch="SM_103"
    )


@pytest.mark.parametrize(
    "metadata, manifest_dtypes",
    [
        ({"capabilities": {"dtype": "float16"}}, ("bfloat16",)),
        (
            {"graph_safe": False, "capabilities": {"graph_mode": "cuda_graph"}},
            (),
        ),
    ],
)
def test_contradictory_variant_domain_rejects_at_registration(
    metadata, manifest_dtypes
):
    eligibility = eligibility_from_metadata(metadata, manifest_dtypes)
    with pytest.raises(VariantRegistrationError, match="empty|contradictory"):
        _registry(eligibility)


def test_inverted_legacy_token_range_rejects_before_registration():
    with pytest.raises(ValueError, match="must not exceed"):
        eligibility_from_metadata(
            {"min_num_tokens": 8, "max_num_tokens": 4}, ()
        )


@pytest.mark.parametrize(
    "metadata, message",
    [
        ({"graph_safe": "false"}, "graph_safe.*boolean"),
        ({"dtypes": "bfloat16"}, "dtypes.*list of strings"),
        ({"architectures": ["sm103", 120]}, "architectures.*non-empty strings"),
        ({"quant": None}, "quant.*list of strings"),
    ],
)
def test_legacy_metadata_schema_fails_closed(metadata, message):
    with pytest.raises(ValueError, match=message):
        eligibility_from_metadata(metadata, ())


@pytest.mark.parametrize(
    "metadata, manifest_dtypes, manifest_architectures, message",
    [
        (
            {"dtypes": ["float16"], "capabilities": {"dtype": "float16"}},
            ("bfloat16",),
            (),
            "dtypes.*disjoint",
        ),
        (
            {"architectures": ["sm120"]},
            (),
            ("sm103",),
            "architectures.*disjoint",
        ),
    ],
)
def test_manifest_and_legacy_metadata_constraints_intersect_fail_closed(
    metadata, manifest_dtypes, manifest_architectures, message
):
    with pytest.raises(ValueError, match=message):
        eligibility_from_metadata(
            metadata, manifest_dtypes, manifest_architectures
        )


@pytest.mark.parametrize("field", ["max_last_dim", "min_num_tokens", "max_num_tokens"])
def test_legacy_numeric_bounds_reject_bool_and_negative(field):
    with pytest.raises(ValueError, match="non-negative integer"):
        eligibility_from_metadata({field: True}, ())
    with pytest.raises(ValueError, match="non-negative integer"):
        eligibility_from_metadata({field: -1}, ())


def test_blockscore_metadata_declares_production_domain_normatively():
    # Keep the contract test self-contained: examples/experiments are intentionally
    # absent from a source release and must not be an implicit pytest dependency.
    metadata = {
        "dtypes": ["bfloat16"],
        "architectures": ["sm103"],
        "graph_safe": False,
        "quant": ["dense"],
        "capabilities": {
            "dtype": "bfloat16",
            "architecture": "sm103",
            "head_dim": 128,
            "block_size": 128,
            "phase": "prefill",
            "layout": "row_major",
            "graph_mode": "eager",
            "quant": "dense",
        },
        "regime": {"description": "prefill sparse block-score selection"},
    }
    domain = capability_domain_from_metadata(metadata)
    exact = {
        predicate.field: predicate.allowed[0]
        for predicate in domain.predicates
        if len(predicate.allowed) == 1
    }

    assert exact.items() >= {
        "dtype": "bfloat16",
        "architecture": "sm103",
        "head_dim": 128,
        "block_size": 128,
        "phase": "prefill",
        "layout": "row_major",
        "graph_mode": "eager",
        "quant": "dense",
    }.items()
    assert "head_dim" not in metadata["regime"]
    assert "block_size" not in metadata["regime"]
