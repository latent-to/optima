"""Shape-specialized implementations sharing one semantic slot."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from optima import receipts
from optima.capabilities import (
    CallDescriptor,
    CapabilityDomain,
    CapabilityPredicate,
)
from optima.manifest import DEFAULT_VARIANT, ManifestError, load_manifest
from optima.registry import (
    Eligibility,
    KernelImpl,
    KernelRegistry,
    SelectionOutcome,
    VariantRegistrationError,
    eligibility_from_metadata,
)


SLOT = "activation.silu_and_mul"


def _write_bundle(
    root: Path,
    rows: list[dict[str, str]],
) -> Path:
    root.mkdir(parents=True)
    lines = ['bundle_id = "variants-test"', 'abi_version = "optima-op-abi-v0"', ""]
    for index, row in enumerate(rows):
        source = row.get("source", f"kernels/k{index}.py")
        entry = row.get("entry", f"entry_{index}")
        lines += ["[[ops]]", f'slot = "{row.get("slot", SLOT)}"']
        if "variant" in row:
            lines.append(f'variant = "{row["variant"]}"')
        lines += [f'source = "{source}"', f'entry = "{entry}"']
        if "metadata" in row:
            lines.append(f'metadata = "{row["metadata"]}"')
        lines.append("")
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"def {entry}(*args):\n    return None\n")
    (root / "manifest.toml").write_text("\n".join(lines))
    return root


def _impl(variant: str, capabilities: dict, *, slot: str = SLOT) -> KernelImpl:
    return KernelImpl(
        slot=slot,
        bundle_id="candidate",
        variant=variant,
        entry=lambda *_args: None,
        eligibility=eligibility_from_metadata(
            {"capabilities": capabilities}, ("bfloat16",)
        ),
    )


def test_legacy_manifest_gets_stable_default_variant(tmp_path):
    manifest = load_manifest(_write_bundle(tmp_path / "bundle", [{}]))

    assert manifest.ops[0].variant == DEFAULT_VARIANT
    assert manifest.op_for(SLOT) is manifest.ops[0]
    assert manifest.ops_for(SLOT) == manifest.ops


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{}, {"variant": "h128"}], "every row.*explicit"),
        (
            [{"variant": "h128"}, {"variant": "h128"}],
            "duplicate variant.*h128",
        ),
        ([{"variant": "bad id"}], "simple identifier"),
    ],
)
def test_duplicate_slot_requires_unique_explicit_variant_ids(tmp_path, rows, message):
    with pytest.raises(ManifestError, match=message):
        load_manifest(_write_bundle(tmp_path / "bundle", rows))


def test_manifest_rejects_non_string_variant(tmp_path):
    bundle = _write_bundle(tmp_path / "bundle", [{}])
    text = (bundle / "manifest.toml").read_text()
    (bundle / "manifest.toml").write_text(
        text.replace('slot = "activation.silu_and_mul"',
                     'slot = "activation.silu_and_mul"\nvariant = 7')
    )

    with pytest.raises(ManifestError, match="variant.*string"):
        load_manifest(bundle)


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("dtypes", '"float16"', "array of strings"),
        ("dtypes", "[16]", "non-empty strings"),
        ("architectures", '"sm120"', "array of strings"),
        ("architectures", '["sm120", ""]', "non-empty strings"),
    ],
)
def test_manifest_eligibility_requires_string_arrays(
    tmp_path, field, value, message
):
    bundle = _write_bundle(tmp_path / "bundle", [{}])
    text = (bundle / "manifest.toml").read_text()
    (bundle / "manifest.toml").write_text(
        text.replace(
            'entry = "entry_0"', f'entry = "entry_0"\n{field} = {value}'
        )
    )

    with pytest.raises(ManifestError, match=message):
        load_manifest(bundle)


def test_manifest_retains_order_and_requires_variant_for_ambiguous_op_for(tmp_path):
    manifest = load_manifest(
        _write_bundle(
            tmp_path / "bundle",
            [{"variant": "h64"}, {"variant": "h128"}],
        )
    )

    assert tuple(op.variant for op in manifest.ops_for(SLOT)) == ("h64", "h128")
    assert manifest.op_for(SLOT, "h128") is manifest.ops[1]
    with pytest.raises(ManifestError, match="multiple variants"):
        manifest.op_for(SLOT)


def test_registry_routes_two_disjoint_shape_variants_and_gaps_to_stock():
    registry = KernelRegistry()
    registry.register(_impl("h64", {"head_dim": 64}))
    registry.register(_impl("h128", {"head_dim": 128}))
    registry.enable()

    base = CallDescriptor(dtype="bfloat16", head_dim=64)
    first = registry.select(SLOT, base, write_fired_receipt=False)
    second = registry.select(
        SLOT, base.with_updates(head_dim=128), write_fired_receipt=False
    )
    gap = registry.select(
        SLOT, base.with_updates(head_dim=96), write_fired_receipt=False
    )

    assert first.impl is registry.variants(SLOT)[0]
    assert first.impl.variant == "h64"
    assert second.impl is registry.variants(SLOT)[1]
    assert second.impl.variant == "h128"
    assert gap.outcome is SelectionOutcome.OUT_OF_DOMAIN
    assert gap.candidate is None
    assert [m.variant for m in gap.variant_matches] == ["h64", "h128"]


def test_manifest_architecture_claim_participates_in_eligibility():
    eligibility = eligibility_from_metadata({}, (), ("sm103",))

    assert eligibility.architectures == frozenset({"sm103"})
    assert eligibility.accepts(dtype_name="bfloat16", last_dim=64, arch="sm103")
    assert not eligibility.accepts(dtype_name="bfloat16", last_dim=64, arch="sm90")


def test_registry_rejects_duplicate_id_and_provable_overlap():
    registry = KernelRegistry()
    registry.register(_impl("small", {"q_len": {"min": 1, "max": 128}}))

    with pytest.raises(VariantRegistrationError, match="duplicate variant"):
        registry.register(_impl("small", {"q_len": 512}))
    with pytest.raises(VariantRegistrationError, match="overlapping capability domains"):
        registry.register(_impl("medium", {"q_len": {"min": 128, "max": 256}}))


def test_selection_fail_closes_runtime_ambiguity_for_future_private_predicate():
    # Public metadata fields are checked at registration.  Programmatic future
    # predicates are deliberately deferred until their semantics are promoted.
    domain = CapabilityDomain(
        (CapabilityPredicate(field="private_shape", allowed=("same",)),)
    )
    registry = KernelRegistry()
    for variant in ("a", "b"):
        registry.register(
            KernelImpl(
                slot=SLOT,
                bundle_id="candidate",
                variant=variant,
                entry=lambda *_args: None,
                eligibility=Eligibility(capabilities=domain),
            )
        )
    registry.enable()

    decision = registry.select(
        SLOT,
        CallDescriptor(private_shape="same"),
        write_fired_receipt=False,
    )

    assert decision.outcome is SelectionOutcome.AMBIGUOUS
    assert decision.impl is None and decision.use_baseline


def test_legacy_lookup_and_peek_stay_compatible_and_fail_closed_on_ambiguity():
    registry = KernelRegistry()
    singleton = KernelImpl(
        slot=SLOT,
        bundle_id="candidate",
        entry=lambda *_args: None,
        eligibility=Eligibility(dtypes=frozenset({"bfloat16"})),
    )
    registry.register(singleton)
    registry.enable()
    kwargs = dict(dtype_name="bfloat16", last_dim=64, arch=None)
    assert registry.peek(SLOT, **kwargs) is singleton
    assert registry.lookup(SLOT, **kwargs) is singleton

    split = KernelRegistry()
    split.register(
        KernelImpl(
            slot=SLOT,
            bundle_id="candidate",
            variant="small",
            entry=lambda *_args: None,
            eligibility=Eligibility(max_num_tokens=31),
        )
    )
    split.register(
        KernelImpl(
            slot=SLOT,
            bundle_id="candidate",
            variant="large",
            entry=lambda *_args: None,
            eligibility=Eligibility(min_num_tokens=32),
        )
    )
    split.enable()
    assert split.peek(SLOT, num_tokens=8, **kwargs).variant == "small"
    assert split.peek(SLOT, num_tokens=64, **kwargs).variant == "large"
    # Historical lookup treats unknown num_tokens as unknown, so both accept it;
    # row order must not become a priority rule.
    assert split.peek(SLOT, num_tokens=None, **kwargs) is None


def test_fired_receipt_is_semantic_slot_level_across_variants(monkeypatch):
    from optima import registry as registry_module

    registry_module._FIRED_SLOTS.discard(SLOT)
    writes: list[tuple[str, dict, str | None]] = []
    monkeypatch.setattr(
        receipts,
        "write",
        lambda event, payload, tag=None: writes.append((event, payload, tag)),
    )
    registry = KernelRegistry()
    registry.register(_impl("h64", {"head_dim": 64}))
    registry.register(_impl("h128", {"head_dim": 128}))
    registry.enable()

    registry.select(SLOT, CallDescriptor(dtype="bfloat16", head_dim=64))
    registry.select(SLOT, CallDescriptor(dtype="bfloat16", head_dim=128))

    assert writes == [("fired", {"slot": SLOT}, SLOT)]
    registry_module._FIRED_SLOTS.discard(SLOT)


def test_seam_loader_registers_every_variant(tmp_path):
    from optima.registry import REGISTRY
    from optima.seam import _load_bundle_into_registry

    bundle = _write_bundle(
        tmp_path / "bundle",
        [
            {"variant": "h64", "metadata": "metadata/h64.json"},
            {"variant": "h128", "metadata": "metadata/h128.json"},
        ],
    )
    metadata = bundle / "metadata"
    metadata.mkdir()
    (metadata / "h64.json").write_text(
        json.dumps({"capabilities": {"head_dim": 64}})
    )
    (metadata / "h128.json").write_text(
        json.dumps({"capabilities": {"head_dim": 128}})
    )

    REGISTRY.clear()
    try:
        _load_bundle_into_registry(str(bundle))
        assert [impl.variant for impl in REGISTRY.variants(SLOT)] == ["h64", "h128"]
    finally:
        REGISTRY.clear()


def test_shared_setup_runs_once_across_variants(tmp_path, monkeypatch):
    from optima import rebuild, sandbox
    from optima.registry import REGISTRY
    from optima.seam import _load_bundle_into_registry

    bundle = tmp_path / "bundle"
    (bundle / "kernels").mkdir(parents=True)
    (bundle / "metadata").mkdir()
    (bundle / "metadata" / "a.json").write_text(
        json.dumps({"capabilities": {"head_dim": 64}})
    )
    (bundle / "metadata" / "b.json").write_text(
        json.dumps({"capabilities": {"head_dim": 128}})
    )
    (bundle / "kernels" / "shared.py").write_text(
        "def entry_a(*args):\n    return None\n"
        "def entry_b(*args):\n    return None\n"
        "def setup_engine():\n    return None\n"
    )
    (bundle / "manifest.toml").write_text(
        'bundle_id = "setup-variants"\n'
        'abi_version = "optima-op-abi-v0"\n'
        '[[ops]]\nslot = "activation.silu_and_mul"\nvariant = "a"\n'
        'source = "kernels/shared.py"\nentry = "entry_a"\nsetup = "setup_engine"\n'
        'metadata = "metadata/a.json"\n'
        '[[ops]]\nslot = "activation.silu_and_mul"\nvariant = "b"\n'
        'source = "kernels/shared.py"\nentry = "entry_b"\nsetup = "setup_engine"\n'
        'metadata = "metadata/b.json"\n'
    )
    calls = []
    load_calls = []
    rebuild_calls = []
    original = sandbox.callable_from
    original_load = sandbox.load_module
    original_rebuild = rebuild.apply_rebuild_plan

    def observed(module, name):
        fn = original(module, name)
        if name != "setup_engine":
            return fn

        def wrapped():
            calls.append(name)
            return fn()

        return wrapped

    def observed_load(path):
        load_calls.append(path)
        return original_load(path)

    def observed_rebuild(path):
        rebuild_calls.append(path)
        return original_rebuild(path)

    monkeypatch.setattr(sandbox, "callable_from", observed)
    monkeypatch.setattr(sandbox, "load_module", observed_load)
    monkeypatch.setattr(rebuild, "apply_rebuild_plan", observed_rebuild)
    REGISTRY.clear()
    try:
        monkeypatch.delenv("OPTIMA_FRAMEWORK_MODE", raising=False)
        with pytest.raises(RuntimeError, match="OPTIMA_FRAMEWORK_MODE is not armed"):
            _load_bundle_into_registry(str(bundle))
        assert load_calls == []
        assert rebuild_calls == []
        assert calls == []

        monkeypatch.setenv("OPTIMA_FRAMEWORK_MODE", "1")
        _load_bundle_into_registry(str(bundle))
        assert len(load_calls) == 1
        assert len(rebuild_calls) == 1
        assert calls == ["setup_engine"]
    finally:
        REGISTRY.clear()
