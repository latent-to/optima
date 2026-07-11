from __future__ import annotations

import difflib
from dataclasses import replace
from pathlib import Path

import pytest

from optima.manifest import (
    ABI_VERSION,
    CompetitionEntry,
    DepPatchEntry,
    Manifest,
    ManifestError,
    load_manifest,
)
from optima.target_catalog import (
    FEATURE_CUDA_SOURCES,
    FEATURE_DEP_PATCH_FLASHINFER,
    FEATURE_ENTRY,
    FEATURE_PREPARE,
    FEATURE_REBUILD_APPLY_DEP_PATCH,
    FEATURE_REBUILD_BUILD_CUDA_EXT,
    FEATURE_SETUP,
    FEATURE_VARIANTS,
    MOE_EPILOGUE_ATOMIC_TARGET,
    MOE_EPILOGUE_MEMBERS,
    SINGLETON_TARGET_IDS,
    ResolvedTarget,
    TargetCatalog,
    TargetCatalogError,
    TargetKind,
    TargetResolutionError,
    TargetSpec,
    default_target_catalog,
    manifest_declared_features,
    resolve_intake_target,
    resolve_target,
)


SILU = "activation.silu_and_mul"


def _diff(path: str = "flashinfer/data/csrc/fused_moe/x.cu") -> str:
    return "".join(
        difflib.unified_diff(
            ["old\n"],
            ["new\n"],
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _bundle(
    tmp_path: Path,
    *,
    rows: tuple[dict[str, object], ...] = ({"slot": SILU},),
    competition: str = "",
    dep_target: str | None = None,
) -> Path:
    root = tmp_path / "bundle"
    root.mkdir(parents=True)
    lines = [
        'bundle_id = "target-test"',
        f'abi_version = "{ABI_VERSION}"',
        "",
    ]
    if competition:
        lines.extend([competition, ""])
    for index, row in enumerate(rows):
        slot = str(row.get("slot", SILU))
        source = str(row.get("source", f"kernels/k{index}.py"))
        entry = str(row.get("entry", f"entry_{index}"))
        source_path = root / source
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text(f"def {entry}(*args):\n    return None\n")
        lines.extend(["[[ops]]", f'slot = "{slot}"'])
        for key in (
            "variant",
            "prepare",
            "setup",
            "base_kernel",
            "override_point",
        ):
            if key in row:
                lines.append(f'{key} = "{row[key]}"')
        lines.extend([f'source = "{source}"', f'entry = "{entry}"'])
        if row.get("cuda_sources"):
            cuda = str(row.get("cuda_path", "kernels/k.cu"))
            cuda_path = root / cuda
            cuda_path.parent.mkdir(parents=True, exist_ok=True)
            cuda_path.write_text("// inspected source\n")
            lines.append(f'cuda_sources = ["{cuda}"]')
        if "extra_key" in row:
            lines.append(f'{row["extra_key"]} = "future"')
        lines.append("")
    if dep_target is not None:
        patch = root / "patches/p.patch"
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text(_diff())
        lines.extend(
            [
                "[[dep_patches]]",
                f'target = "{dep_target}"',
                'path = "patches/p.patch"',
                "",
            ]
        )
    (root / "manifest.toml").write_text("\n".join(lines))
    return root


def _competition(target: str, mode: str) -> str:
    return f'[competition]\ntarget = "{target}"\nmode = "{mode}"'


def _slot_spec(
    target_id: str,
    *,
    displaces: frozenset[str] = frozenset(),
    compatible_with: frozenset[str] = frozenset(),
    features: frozenset[str] = frozenset({FEATURE_ENTRY}),
) -> TargetSpec:
    return TargetSpec(
        target_id=target_id,
        kind=TargetKind.SLOT,
        members=(target_id,),
        displaces=displaces,
        compatible_with=compatible_with,
        allowed_features=features,
    )


def _atomic_spec(
    target_id: str = "atomic.ab",
    *,
    members: tuple[str, ...] = ("slot.a", "slot.b"),
    displaces: frozenset[str] | None = None,
) -> TargetSpec:
    return TargetSpec(
        target_id=target_id,
        kind=TargetKind.ATOMIC,
        members=members,
        displaces=frozenset(members) if displaces is None else displaces,
        allowed_features=frozenset({FEATURE_ENTRY}),
    )


# -- syntax-only manifest request -------------------------------------------


def test_manifest_parses_syntax_only_competition_request(tmp_path):
    manifest = load_manifest(
        _bundle(tmp_path, competition=_competition(SILU, "slot"))
    )

    assert manifest.competition == CompetitionEntry(target=SILU, mode="slot")


@pytest.mark.parametrize(
    "table, message",
    [
        ("competition = []", "must be a .* table"),
        (
            '[competition]\ntarget = "activation.silu_and_mul"\n'
            'mode = "slot"\nmembers = ["miner"]',
            "unknown keys",
        ),
        ('[competition]\nmode = "slot"', "target.*string"),
        ('[competition]\ntarget = 7\nmode = "slot"', "target.*string"),
        ('[competition]\ntarget = "bad id"\nmode = "slot"', "simple identifier"),
        (f'[competition]\ntarget = "{SILU}"', "mode.*string"),
        (f'[competition]\ntarget = "{SILU}"\nmode = 7', "mode.*string"),
        (
            f'[competition]\ntarget = "{SILU}"\nmode = "per_slot"',
            "slot.*atomic",
        ),
    ],
)
def test_manifest_rejects_malformed_competition(tmp_path, table, message):
    with pytest.raises(ManifestError, match=message):
        load_manifest(_bundle(tmp_path, competition=table))


def test_legacy_system_request_parses_but_never_registers_a_title(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            competition=_competition("sglang.inference.bundle.v1", "system"),
        )
    )

    assert manifest.competition == CompetitionEntry(
        target="sglang.inference.bundle.v1", mode="system"
    )
    resolved = resolve_target(manifest)
    assert not resolved.registered and not resolved.implicit
    assert resolved.target_id is None
    assert "legacy competition mode 'system'" in (resolved.reason or "")
    with pytest.raises(TargetResolutionError, match="legacy competition mode"):
        resolve_intake_target(manifest, observed_features=())


def test_manifest_field_append_preserves_historical_positional_arguments():
    patch = DepPatchEntry(target="flashinfer", path="p.patch")
    manifest = Manifest("bundle", ABI_VERSION, (), (patch,), {"old": "raw"})

    assert manifest.dep_patches == (patch,)
    assert manifest.raw == {"old": "raw"}
    assert manifest.competition is None


def test_all_tracked_examples_remain_implicit_and_parseable():
    examples = Path(__file__).resolve().parents[1] / "examples"
    manifests = sorted(examples.glob("*/manifest.toml"))
    assert manifests
    for path in manifests:
        assert load_manifest(path.parent).competition is None


# -- canonical resolution ---------------------------------------------------


def test_implicit_and_explicit_singleton_resolve_to_catalog_identity(tmp_path):
    implicit = resolve_target(load_manifest(_bundle(tmp_path / "implicit")))
    explicit = resolve_target(
        load_manifest(
            _bundle(
                tmp_path / "explicit",
                competition=_competition(SILU, "slot"),
            )
        )
    )

    assert implicit == ResolvedTarget(
        target_id=SILU,
        kind=TargetKind.SLOT,
        members=(SILU,),
        registered=True,
        implicit=True,
        observed_features=frozenset({FEATURE_ENTRY}),
        features_complete=False,
    )
    assert explicit.target_id == SILU
    assert explicit.members == (SILU,)
    assert not explicit.implicit


def test_multiple_variants_are_one_semantic_member(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=(
                {"slot": SILU, "variant": "small"},
                {"slot": SILU, "variant": "large"},
            ),
        )
    )

    resolved = resolve_target(manifest)
    assert resolved.members == (SILU,)
    assert resolved.observed_features == frozenset(
        {FEATURE_ENTRY, FEATURE_VARIANTS}
    )


def test_atomic_resolution_uses_catalog_order_not_manifest_or_variant_order(tmp_path):
    first, second = MOE_EPILOGUE_MEMBERS
    manifest = load_manifest(
        _bundle(
            tmp_path,
            competition=_competition(MOE_EPILOGUE_ATOMIC_TARGET, "atomic"),
            rows=(
                {"slot": second},
                {"slot": first, "variant": "large"},
                {"slot": first, "variant": "small"},
            ),
        )
    )

    resolved = resolve_target(manifest)
    assert resolved.target_id == MOE_EPILOGUE_ATOMIC_TARGET
    assert resolved.members == MOE_EPILOGUE_MEMBERS
    assert resolved.observed_features >= {FEATURE_ENTRY, FEATURE_VARIANTS}


def test_legacy_exact_atomic_pair_resolves_implicitly(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=tuple({"slot": member} for member in reversed(MOE_EPILOGUE_MEMBERS)),
        )
    )

    resolved = resolve_target(manifest)
    assert resolved.target_id == MOE_EPILOGUE_ATOMIC_TARGET
    assert resolved.members == MOE_EPILOGUE_MEMBERS
    assert resolved.implicit


@pytest.mark.parametrize(
    "competition, rows, message",
    [
        (_competition("unknown.target", "slot"), ({"slot": SILU},), "unknown"),
        (
            _competition(SILU, "atomic"),
            ({"slot": SILU},),
            "catalog kind.*not requested",
        ),
        (
            _competition(MOE_EPILOGUE_ATOMIC_TARGET, "atomic"),
            ({"slot": MOE_EPILOGUE_MEMBERS[0]},),
            "requires exact members",
        ),
        (
            _competition(MOE_EPILOGUE_ATOMIC_TARGET, "atomic"),
            tuple({"slot": member} for member in MOE_EPILOGUE_MEMBERS)
            + ({"slot": SILU},),
            "requires exact members",
        ),
        (
            _competition(SILU, "slot"),
            ({"slot": "norm.rmsnorm"},),
            "requires exact members",
        ),
    ],
)
def test_explicit_target_request_fails_closed(tmp_path, competition, rows, message):
    manifest = load_manifest(
        _bundle(tmp_path, competition=competition, rows=rows)
    )
    with pytest.raises(TargetResolutionError, match=message):
        resolve_target(manifest)


def test_programmatic_invalid_mode_still_fails_closed(tmp_path):
    manifest = replace(
        load_manifest(_bundle(tmp_path)),
        competition=CompetitionEntry(target=SILU, mode="per_slot"),
    )
    with pytest.raises(TargetResolutionError, match="unknown competition mode"):
        resolve_target(manifest)


@pytest.mark.parametrize(
    "competition_value, message",
    [
        (object(), "must be a CompetitionEntry"),
        (CompetitionEntry(target=[], mode="slot"), "malformed"),  # type: ignore[arg-type]
        (CompetitionEntry(target=SILU, mode=7), "malformed"),  # type: ignore[arg-type]
    ],
)
def test_programmatic_malformed_request_fails_closed(
    tmp_path, competition_value, message
):
    manifest = replace(
        load_manifest(_bundle(tmp_path)), competition=competition_value
    )
    with pytest.raises(TargetResolutionError, match=message):
        resolve_target(manifest)


def test_unknown_implicit_multi_op_routes_to_discovery(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=({"slot": SILU}, {"slot": "norm.rmsnorm"}),
        )
    )

    resolved = resolve_target(manifest)
    assert not resolved.registered
    assert resolved.target_id is None and resolved.kind is None
    assert resolved.members == (SILU, "norm.rmsnorm")
    assert "future discovery" in (resolved.reason or "")
    with pytest.raises(TargetResolutionError, match="discovery"):
        resolved.require_registered()
    with pytest.raises(TargetResolutionError, match="future discovery"):
        resolve_intake_target(manifest, observed_features=())


# -- validator catalog invariants ------------------------------------------


def test_default_catalog_has_exactly_one_singleton_per_live_slot():
    from optima.slots import SLOTS

    catalog = default_target_catalog()
    assert set(SINGLETON_TARGET_IDS) == set(SLOTS)
    for target_id in (*SINGLETON_TARGET_IDS, MOE_EPILOGUE_ATOMIC_TARGET):
        assert catalog.require(target_id).target_id == target_id


@pytest.mark.parametrize(
    "specs, message",
    [
        ([], "must not be empty"),
        ({"slot.a": _slot_spec("slot.a")}, "iterable of TargetSpec"),
        ([_slot_spec("slot.a"), _slot_spec("slot.a")], "duplicate target ID"),
        (
            [TargetSpec("slot.a", TargetKind.SLOT, ())],
            "non-empty sequence",
        ),
        (
            [TargetSpec("slot.a", TargetKind.SLOT, ("slot.a", "slot.a"))],
            "duplicate members",
        ),
        ([_slot_spec("slot.a"), _atomic_spec(members=("slot.a",))], "at least two"),
        (
            [
                _slot_spec("slot.a"),
                _slot_spec("slot.b"),
                _atomic_spec(displaces=frozenset({"slot.a"})),
            ],
            "explicitly displace",
        ),
        (
            [
                _slot_spec("slot.a"),
                _atomic_spec(displaces=frozenset({"slot.a"})),
            ],
            "without registered singletons",
        ),
        (
            [TargetSpec("alias", TargetKind.SLOT, ("slot.a",))],
            "itself as its sole member",
        ),
        (
            [_slot_spec("slot.a", displaces=frozenset({"missing"}))],
            "unknown targets",
        ),
        (
            [_slot_spec("slot.a", displaces=frozenset({"slot.a"}))],
            "itself",
        ),
        (
            [_slot_spec("slot.a", features=frozenset({FEATURE_ENTRY, "future"}))],
            "unknown features",
        ),
        (
            [_slot_spec("slot.a", features=frozenset({FEATURE_ENTRY, FEATURE_SETUP}))],
            "may not allow.*setup",
        ),
        ([_slot_spec("slot.a", features=frozenset())], "must allow the entry"),
    ],
)
def test_catalog_rejects_invalid_validator_policy(specs, message):
    with pytest.raises(TargetCatalogError, match=message):
        TargetCatalog(specs)


def test_catalog_rejects_duplicate_atomic_member_sets():
    specs = [
        _slot_spec("slot.a"),
        _slot_spec("slot.b"),
        _atomic_spec("atomic.ab"),
        _atomic_spec("atomic.alias", members=("slot.b", "slot.a")),
    ]
    with pytest.raises(TargetCatalogError, match="same exact member set"):
        TargetCatalog(specs)


def test_partial_atomic_overlap_requires_an_explicit_relationship():
    singletons = [_slot_spec(f"slot.{name}") for name in ("a", "b", "c")]
    atomic_ab = _atomic_spec("atomic.ab", members=("slot.a", "slot.b"))
    atomic_bc = _atomic_spec("atomic.bc", members=("slot.b", "slot.c"))

    with pytest.raises(TargetCatalogError, match="share members.*explicit"):
        TargetCatalog([*singletons, atomic_ab, atomic_bc])

    atomic_ab = replace(
        atomic_ab, compatible_with=frozenset({"atomic.bc"})
    )
    atomic_bc = replace(
        atomic_bc, compatible_with=frozenset({"atomic.ab"})
    )
    catalog = TargetCatalog([*singletons, atomic_ab, atomic_bc])
    assert catalog.validate_active_targets(("atomic.ab", "atomic.bc")) == (
        "atomic.ab",
        "atomic.bc",
    )


def test_compatible_overlap_must_be_symmetric_and_not_displaced():
    with pytest.raises(TargetCatalogError, match="must be symmetric"):
        TargetCatalog(
            [
                _slot_spec("slot.a", compatible_with=frozenset({"slot.b"})),
                _slot_spec("slot.b"),
            ]
        )
    with pytest.raises(TargetCatalogError, match="both displaces and is compatible"):
        TargetCatalog(
            [
                _slot_spec(
                    "slot.a",
                    displaces=frozenset({"slot.b"}),
                    compatible_with=frozenset({"slot.b"}),
                ),
                _slot_spec("slot.b", compatible_with=frozenset({"slot.a"})),
            ]
        )


def test_catalog_rejects_displacement_cycle():
    with pytest.raises(TargetCatalogError, match="cycle"):
        TargetCatalog(
            [
                _slot_spec("slot.a", displaces=frozenset({"slot.b"})),
                _slot_spec("slot.b", displaces=frozenset({"slot.a"})),
            ]
        )


def test_catalog_registration_order_does_not_change_resolution():
    a = _slot_spec("slot.a")
    b = _slot_spec("slot.b")
    source = [b, a]
    first = TargetCatalog(source)
    source.clear()
    second = TargetCatalog([a, b])
    assert first.require("slot.a") == second.require("slot.a") == a
    assert first.require("slot.b") == second.require("slot.b") == b


def test_default_displacement_and_compatible_overlap_are_explicit():
    catalog = default_target_catalog()
    atomic = catalog.require(MOE_EPILOGUE_ATOMIC_TARGET)

    assert atomic.displaces == frozenset(MOE_EPILOGUE_MEMBERS)
    assert catalog.require("moe.fused_experts").compatible_with == frozenset(
        {"moe.fused_experts_reduce"}
    )
    assert catalog.validate_active_targets(
        ["moe.fused_experts", "moe.fused_experts_reduce"]
    ) == ("moe.fused_experts", "moe.fused_experts_reduce")
    with pytest.raises(TargetResolutionError, match="displaces"):
        catalog.validate_active_targets(
            [MOE_EPILOGUE_MEMBERS[0], MOE_EPILOGUE_ATOMIC_TARGET]
        )

    with pytest.raises(TargetResolutionError, match="must be strings"):
        catalog.validate_active_targets((["unhashable"],))  # type: ignore[list-item]


# -- contribution feature admission ---------------------------------------


def test_standard_target_admits_variants_prepare_override_and_cuda(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=(
                {
                    "slot": "moe.fused_experts",
                    "variant": "sm120",
                    "prepare": "prepare",
                    "base_kernel": "nvfp4_moe",
                    "override_point": "epilogue",
                    "cuda_sources": True,
                },
            ),
        )
    )

    resolved = resolve_target(manifest)
    assert resolved.registered
    assert not resolved.features_complete
    with pytest.raises(TargetResolutionError, match="lacks complete"):
        resolved.require_complete_features()
    assert manifest_declared_features(manifest) >= {
        FEATURE_ENTRY,
        FEATURE_VARIANTS,
        FEATURE_PREPARE,
        FEATURE_CUDA_SOURCES,
    }


def test_setup_manifest_parses_but_registered_target_rejects_it(tmp_path):
    manifest = load_manifest(
        _bundle(tmp_path, rows=({"slot": SILU, "setup": "setup"},))
    )
    assert FEATURE_SETUP in manifest_declared_features(manifest)
    with pytest.raises(TargetResolutionError, match="fenced discovery lane"):
        resolve_target(manifest)


def test_unknown_op_extension_cannot_bypass_feature_policy(tmp_path):
    manifest = load_manifest(
        _bundle(tmp_path, rows=({"slot": SILU, "extra_key": "future_knob"},))
    )
    with pytest.raises(TargetResolutionError, match="op_extra:future_knob"):
        resolve_target(manifest)


def test_shallow_native_bundle_admits_exact_reviewed_builder(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=(
                {
                    "slot": "collective.ar_residual_rmsnorm",
                    "cuda_sources": True,
                },
            ),
            competition=_competition("collective.ar_residual_rmsnorm", "slot"),
        )
    )

    assert FEATURE_REBUILD_BUILD_CUDA_EXT not in manifest_declared_features(manifest)
    resolved = resolve_intake_target(
        manifest, observed_features=(FEATURE_REBUILD_BUILD_CUDA_EXT,)
    )
    assert resolved.registered and resolved.features_complete
    assert resolved.observed_features >= {
        FEATURE_CUDA_SOURCES,
        FEATURE_REBUILD_BUILD_CUDA_EXT,
    }


def test_deep_atomic_admits_only_exact_flashinfer_patch_lane(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=tuple(
                {"slot": member, "cuda_sources": True}
                for member in MOE_EPILOGUE_MEMBERS
            ),
            competition=_competition(MOE_EPILOGUE_ATOMIC_TARGET, "atomic"),
            dep_target="flashinfer",
        )
    )

    resolved = resolve_intake_target(
        manifest,
        observed_features=(
            FEATURE_REBUILD_APPLY_DEP_PATCH,
            FEATURE_REBUILD_BUILD_CUDA_EXT,
        ),
    )
    assert resolved.registered
    assert resolved.observed_features >= {
        FEATURE_DEP_PATCH_FLASHINFER,
        FEATURE_REBUILD_APPLY_DEP_PATCH,
    }


def test_deep_singleton_admits_its_exact_flashinfer_capability(tmp_path):
    target = "collective.moe_finalize_ar_rmsnorm"
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=({"slot": target, "cuda_sources": True},),
            competition=_competition(target, "slot"),
            dep_target="flashinfer",
        )
    )
    resolved = resolve_intake_target(
        manifest,
        observed_features=(
            FEATURE_REBUILD_APPLY_DEP_PATCH,
            FEATURE_REBUILD_BUILD_CUDA_EXT,
        ),
    )
    assert resolved.target_id == target and resolved.features_complete


def test_unrelated_dependency_patch_is_not_registered_for_deep_target(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=tuple({"slot": member} for member in MOE_EPILOGUE_MEMBERS),
            competition=_competition(MOE_EPILOGUE_ATOMIC_TARGET, "atomic"),
            dep_target="leftpad",
        )
    )
    with pytest.raises(TargetResolutionError, match="dep_patch:leftpad"):
        resolve_intake_target(
            manifest, observed_features=(FEATURE_REBUILD_APPLY_DEP_PATCH,)
        )


def test_exact_observed_rebuild_capability_is_target_policy_not_manifest_data(tmp_path):
    manifest = load_manifest(_bundle(tmp_path))
    assert FEATURE_REBUILD_BUILD_CUDA_EXT not in manifest_declared_features(manifest)
    with pytest.raises(TargetResolutionError, match="rebuild:apply_dep_patch"):
        resolve_intake_target(
            manifest, observed_features=(FEATURE_REBUILD_APPLY_DEP_PATCH,)
        )
    with pytest.raises(TargetResolutionError, match="rebuild:unknown"):
        resolve_intake_target(manifest, observed_features=("rebuild:unknown",))


def test_complete_feature_evidence_binds_patch_declaration_to_applier(tmp_path):
    manifest = load_manifest(
        _bundle(
            tmp_path,
            rows=tuple({"slot": member} for member in MOE_EPILOGUE_MEMBERS),
            competition=_competition(MOE_EPILOGUE_ATOMIC_TARGET, "atomic"),
            dep_target="flashinfer",
        )
    )
    with pytest.raises(TargetResolutionError, match="patch without.*apply_dep_patch"):
        resolve_intake_target(manifest, observed_features=())

    no_patch = load_manifest(
        _bundle(
            tmp_path / "no_patch",
            rows=tuple({"slot": member} for member in MOE_EPILOGUE_MEMBERS),
            competition=_competition(MOE_EPILOGUE_ATOMIC_TARGET, "atomic"),
        )
    )
    with pytest.raises(TargetResolutionError, match="without a declared"):
        resolve_intake_target(
            no_patch, observed_features=(FEATURE_REBUILD_APPLY_DEP_PATCH,)
        )


def test_observed_features_argument_is_strict(tmp_path):
    manifest = load_manifest(_bundle(tmp_path))
    with pytest.raises(TargetResolutionError, match="iterable"):
        default_target_catalog().resolve_intake(
            manifest, observed_features="rebuild:build_cuda_ext"
        )
    with pytest.raises(TargetResolutionError, match="non-empty strings"):
        default_target_catalog().resolve_intake(
            manifest, observed_features=("",)
        )


def test_target_catalog_has_no_economic_or_trust_policy_imports():
    source = (
        Path(__file__).resolve().parents[1] / "optima/target_catalog.py"
    ).read_text()
    forbidden = (
        "optima.chain",
        "optima.commit_reveal",
        "optima.device_component",
        "optima.system_patch",
        "crownable",
        "for_settlement",
    )
    assert not [token for token in forbidden if token in source]
