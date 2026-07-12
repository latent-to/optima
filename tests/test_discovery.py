from __future__ import annotations

import difflib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from optima.compat import PINNED_SGLANG
from optima.discovery import (
    DEFAULT_DISCOVERY_POLICY,
    DISCOVERY_ABI_VERSION,
    DiscoveryArmPlan,
    DiscoveryBuildProfile,
    DiscoveryError,
    DiscoveryPromotion,
    DiscoveryWinRecord,
    build_discovery_overlay_stage,
    discovery_candidate_stack_digest,
    discovery_selected_delta_digest,
    inspect_discovery,
    load_discovery_manifest,
    reopen_discovery_overlay,
    require_discovery_build_profile,
    validate_discovery_file_patch,
    validate_discovery_patch_path,
    validate_discovery_patch_set,
)
from optima.deppatch import parse_patch_text
from optima.eval.native_artifact import publish_native_artifact
from optima.manifest import ManifestError, load_manifest
from optima.stack_identity import sha256_hex
from optima.stack_manifest import EvaluationStackManifest
from optima.target_catalog import default_target_catalog


ARENA = "minimax-m3-rtx-tp8-v1"
MODEL = "minimax-m3-nvfp4"
PROFILE_ID = "minimax-m3-rtx-sm120-tp8-v1"


def _h(label: str) -> str:
    return sha256_hex(label.encode())


def _diff(old: str, new: str, path: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )


def _manifest(
    *,
    abi: str = DISCOVERY_ABI_VERSION,
    patches: tuple[str, ...] = ("patches/change.patch",),
    dependencies: tuple[str, ...] = ("cuda13",),
    conflicts: tuple[str, ...] = (),
    profile: str = PROFILE_ID,
    promotion: str = "new_singleton",
    extra: str = "",
) -> str:
    quoted_patches = ", ".join(json.dumps(value) for value in patches)
    quoted_dependencies = ", ".join(json.dumps(value) for value in dependencies)
    quoted_conflicts = ", ".join(json.dumps(value) for value in conflicts)
    return (
        'bundle_id = "proposal-one"\n'
        f"abi_version = {json.dumps(abi)}\n"
        f"build_profile = {json.dumps(profile)}\n"
        f"patches = [{quoted_patches}]\n"
        f"dependencies = [{quoted_dependencies}]\n"
        f"conflicts = [{quoted_conflicts}]\n"
        f"requested_promotion = {json.dumps(promotion)}\n"
        f"{extra}"
        "\n[applicability]\n"
        f'arenas = ["{ARENA}"]\n'
        f'models = ["{MODEL}"]\n'
        'architectures = ["sm120"]\n'
        "tensor_parallel_sizes = [8]\n"
    )


def _bundle(
    tmp_path: Path,
    *,
    patch_text: str | None = None,
    manifest_text: str | None = None,
    name: str = "bundle",
) -> Path:
    root = tmp_path / name
    (root / "patches").mkdir(parents=True)
    (root / "manifest.toml").write_text(manifest_text or _manifest())
    (root / "patches/change.patch").write_text(
        patch_text
        or _diff(
            "VALUE = 1\n",
            "VALUE = 2\n",
            "sglang/srt/layers/activation.py",
        )
    )
    return root


def _stock(tmp_path: Path, *, name: str = "stock") -> Path:
    site = tmp_path / name
    package = site / "sglang"
    (package / "srt/layers").mkdir(parents=True)
    (package / "srt/managers").mkdir(parents=True)
    (package / "srt/layers/activation.py").write_text("VALUE = 1\n")
    (package / "untouched.py").write_text("UNCHANGED = True\n")
    (package / "srt/managers/scheduler.py").write_text(
        "class Scheduler:\n"
        "    def get_next_batch_to_run(self):\n"
        "        choice = 1\n"
        "        return choice\n"
        "\n"
        "    def process_batch_result(self):\n"
        "        result = 1\n"
        "        return result\n"
    )
    return site


def _profile(
    *,
    features: tuple[str, ...] = ("cuda13",),
    architecture: str = "sm120",
    profile_id: str = PROFILE_ID,
) -> DiscoveryBuildProfile:
    return DiscoveryBuildProfile(
        profile_id=profile_id,
        sglang_version=DEFAULT_DISCOVERY_POLICY.sglang_version,
        arena=ARENA,
        model=MODEL,
        architecture=architecture,
        tensor_parallel_size=8,
        features=features,
        build_inputs=(("image", _h("image")), ("worker", _h("worker"))),
    )


def _incumbent() -> EvaluationStackManifest:
    catalog = default_target_catalog()
    return EvaluationStackManifest(
        runtime_digest=_h("runtime"),
        base_engine_digest=_h("base-engine"),
        arena_digest=_h("arena"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={},
    )


def test_default_discovery_policy_tracks_the_canonical_sglang_pin() -> None:
    assert DEFAULT_DISCOVERY_POLICY.sglang_version == PINNED_SGLANG


def test_discovery_abi_is_separate_from_component_manifest(tmp_path):
    root = _bundle(tmp_path)
    manifest = load_discovery_manifest(root)
    assert manifest.abi_version == DISCOVERY_ABI_VERSION
    assert manifest.bundle_id == "proposal-one"
    assert manifest.requested_promotion == "new_singleton"
    assert manifest.applicability.tensor_parallel_sizes == (8,)

    with pytest.raises(ManifestError, match="unsupported abi_version"):
        load_manifest(root)


@pytest.mark.parametrize(
    "manifest_text,match",
    [
        (_manifest(abi="optima-op-abi-v0"), "unsupported abi_version"),
        (_manifest(extra='command = "bash evil.sh"\n'), "schema mismatch"),
        (_manifest(extra='environment = { HOME = "/tmp" }\n'), "schema mismatch"),
        (_manifest(patches=()), "patches must be a nonempty"),
        (_manifest(patches=("../escape.patch",)), "canonical relative path"),
        (_manifest(patches=("patches/change.txt",)), "must use .patch"),
        (_manifest(dependencies=("z", "a")), "sorted and unique"),
        (
            _manifest(dependencies=("cuda13",), conflicts=("cuda13",)),
            "must be disjoint",
        ),
        (_manifest(promotion="ship_everywhere"), "requested_promotion"),
    ],
)
def test_manifest_is_closed_and_canonical(tmp_path, manifest_text, match):
    root = _bundle(tmp_path, manifest_text=manifest_text)
    with pytest.raises(DiscoveryError, match=match):
        load_discovery_manifest(root)


@pytest.mark.parametrize("field", ("patches", "requested_promotion"))
def test_hostile_toml_scalar_types_fail_typed(tmp_path, field):
    text = _manifest()
    old = 'patches = ["patches/change.patch"]' if field == "patches" else 'requested_promotion = "new_singleton"'
    root = _bundle(tmp_path, manifest_text=text.replace(old, f"{field} = 1"))
    with pytest.raises(DiscoveryError):
        load_discovery_manifest(root)


def test_inspection_freezes_exact_inventory_and_patch_bytes(tmp_path):
    root = _bundle(tmp_path)
    inspected = inspect_discovery(root)
    assert [row.path for row in inspected.files] == [
        "manifest.toml",
        "patches/change.patch",
    ]
    assert inspected.patch_texts[0][0] == "patches/change.patch"
    assert len(inspected.proposal_digest) == 64

    revised = _bundle(tmp_path, name="revised")
    text = (revised / "manifest.toml").read_text()
    (revised / "manifest.toml").write_text(text + "\n")
    assert inspect_discovery(revised).proposal_digest != inspected.proposal_digest

    with pytest.raises(DiscoveryError, match="differ from inventory"):
        replace(
            inspected,
            patch_texts=(("patches/change.patch", "malicious replacement\n"),),
        )


def test_inspection_rejects_undeclared_files_directories_and_symlinks(tmp_path):
    extra_file = _bundle(tmp_path, name="extra-file")
    (extra_file / "notes.txt").write_text("not declared")
    with pytest.raises(DiscoveryError, match="file inventory differs"):
        inspect_discovery(extra_file)

    extra_dir = _bundle(tmp_path, name="extra-dir")
    (extra_dir / "empty").mkdir()
    with pytest.raises(DiscoveryError, match="undeclared directory"):
        inspect_discovery(extra_dir)

    symlink = _bundle(tmp_path, name="symlink")
    target = symlink / "real.patch"
    target.write_text((symlink / "patches/change.patch").read_text())
    (symlink / "patches/change.patch").unlink()
    (symlink / "patches/change.patch").symlink_to(target)
    with pytest.raises(DiscoveryError, match="regular single-linked"):
        inspect_discovery(symlink)


@pytest.mark.parametrize(
    "path",
    [
        "sglang/srt/layers/activation.py",
        "sglang/srt/model_executor/model_runner.py",
        "sglang/srt/models/minimax.py",
        "sglang/srt/mem_cache/memory_pool.py",
        "sglang/srt/batch_overlap/two_batch_overlap.py",
        "sglang/srt/managers/schedule_policy.py",
        "sglang/srt/managers/scheduler.py",
    ],
)
def test_policy_admits_data_plane_source(path):
    validate_discovery_patch_path(DEFAULT_DISCOVERY_POLICY, path)


@pytest.mark.parametrize(
    "path",
    [
        "../sglang/srt/layers/activation.py",
        "sglang/api.py",
        "sglang/srt/entrypoints/engine.py",
        "sglang/srt/managers/tokenizer_manager.py",
        "sglang/srt/managers/detokenizer_manager.py",
        "sglang/srt/managers/scheduler_components/logprob_result_processor.py",
        "sglang/srt/managers/scheduler_components/batch_result_processor.py",
        "sglang/srt/layers/sampler.py",
        "sglang/srt/layers/logits_processor.py",
        "sglang/srt/observability/timing.py",
        "sglang/srt/layers/kernel.so",
    ],
)
def test_policy_excludes_control_and_grading_surfaces(path):
    with pytest.raises(DiscoveryError):
        validate_discovery_patch_path(DEFAULT_DISCOVERY_POLICY, path)


def test_allowed_path_cannot_add_an_excluded_import():
    patch = _diff(
        "VALUE = 1\n",
        "from sglang.srt.layers.logits_processor import LogitsProcessor\nVALUE = 1\n",
        "sglang/srt/layers/activation.py",
    )
    (file_patch,) = parse_patch_text(patch)
    with pytest.raises(DiscoveryError, match="excluded source"):
        validate_discovery_file_patch(
            DEFAULT_DISCOVERY_POLICY,
            file_patch,
            original_source="VALUE = 1\n",
        )


@pytest.mark.parametrize(
    "original,addition",
    [
        ("VALUE = 1\n", "from . import sampler\n"),
        ("VALUE = 1\n", 'import importlib\nimportlib.import_module("samp" + "ler")\n'),
        (
            "from sglang.srt.layers.sampler import Sampler as S\nVALUE = 1\n",
            "USE = S\n",
        ),
    ],
)
def test_excluded_source_cannot_use_relative_dynamic_or_existing_alias(original, addition):
    (file_patch,) = parse_patch_text(
        _diff(original, original + addition, "sglang/srt/layers/activation.py")
    )
    with pytest.raises(DiscoveryError, match="excluded/dynamic"):
        validate_discovery_file_patch(
            DEFAULT_DISCOVERY_POLICY, file_patch, original_source=original
        )


def test_mixed_scheduler_file_is_limited_to_named_symbols():
    original = (
        "class Scheduler:\n"
        "    def get_next_batch_to_run(self):\n"
        "        choice = 1\n"
        "        return choice\n"
        "\n"
        "    def process_batch_result(self):\n"
        "        result = 1\n"
        "        return result\n"
    )
    allowed = original.replace("choice = 1", "choice = 2")
    (allowed_patch,) = parse_patch_text(
        _diff(original, allowed, "sglang/srt/managers/scheduler.py")
    )
    validate_discovery_file_patch(
        DEFAULT_DISCOVERY_POLICY, allowed_patch, original_source=original
    )

    forbidden = original.replace("result = 1", "result = 2")
    (forbidden_patch,) = parse_patch_text(
        _diff(original, forbidden, "sglang/srt/managers/scheduler.py")
    )
    with pytest.raises(DiscoveryError, match="outside allowed symbols"):
        validate_discovery_file_patch(
            DEFAULT_DISCOVERY_POLICY, forbidden_patch, original_source=original
        )

    escaped = original.replace(
        "        return choice\n\n",
        "        return choice\n\n    ESCAPED_CLASS_LEVEL = True\n\n",
    )
    (escaped_patch,) = parse_patch_text(
        _diff(original, escaped, "sglang/srt/managers/scheduler.py")
    )
    with pytest.raises(DiscoveryError, match="outside allowed symbols"):
        validate_discovery_file_patch(
            DEFAULT_DISCOVERY_POLICY, escaped_patch, original_source=original
        )

    lexical = original.replace(
        "        choice = 1\n",
        '        choice = 1\n        """\n',
    ).replace("        return result\n", '        return result\n        """\n')
    (lexical_patch,) = parse_patch_text(
        _diff(original, lexical, "sglang/srt/managers/scheduler.py")
    )
    with pytest.raises(DiscoveryError, match="outside allowed symbols"):
        validate_discovery_file_patch(
            DEFAULT_DISCOVERY_POLICY, lexical_patch, original_source=original
        )


def test_build_profile_is_validator_owned_and_exact(tmp_path):
    manifest = inspect_discovery(_bundle(tmp_path)).manifest
    require_discovery_build_profile(
        manifest, _profile(), DEFAULT_DISCOVERY_POLICY
    )

    with pytest.raises(DiscoveryError, match="another validator build profile"):
        require_discovery_build_profile(
            manifest,
            _profile(profile_id="other-profile"),
            DEFAULT_DISCOVERY_POLICY,
        )
    with pytest.raises(DiscoveryError, match="outside proposal applicability"):
        require_discovery_build_profile(
            manifest,
            _profile(architecture="sm103"),
            DEFAULT_DISCOVERY_POLICY,
        )
    with pytest.raises(DiscoveryError, match="lacks proposal dependencies"):
        require_discovery_build_profile(
            manifest,
            _profile(features=()),
            DEFAULT_DISCOVERY_POLICY,
        )


def test_profile_conflicts_fail_before_patch_application(tmp_path):
    root = _bundle(
        tmp_path,
        manifest_text=_manifest(dependencies=(), conflicts=("fa4",)),
    )
    manifest = inspect_discovery(root).manifest
    with pytest.raises(DiscoveryError, match="activates proposal conflicts"):
        require_discovery_build_profile(
            manifest,
            _profile(features=("fa4",)),
            DEFAULT_DISCOVERY_POLICY,
        )


def test_patch_set_binds_stock_tree_and_rejects_duplicate_targets(tmp_path):
    stock = _stock(tmp_path)
    inspected = inspect_discovery(_bundle(tmp_path))
    validated = validate_discovery_patch_set(
        inspected, DEFAULT_DISCOVERY_POLICY, stock
    )
    assert validated.touched_paths == ("sglang/srt/layers/activation.py",)
    assert len(validated.stock_tree_digest) == 64

    duplicate = _bundle(
        tmp_path,
        name="duplicate",
        manifest_text=_manifest(
            patches=("patches/a.patch", "patches/b.patch")
        ),
    )
    common = (duplicate / "patches/change.patch").read_text()
    (duplicate / "patches/change.patch").unlink()
    (duplicate / "patches/a.patch").write_text(common)
    (duplicate / "patches/b.patch").write_text(common)
    with pytest.raises(DiscoveryError, match="touched more than once"):
        validate_discovery_patch_set(
            inspect_discovery(duplicate), DEFAULT_DISCOVERY_POLICY, stock
        )


def test_overlay_stage_is_fixed_and_reopens_from_native_publication(tmp_path):
    bundle = _bundle(tmp_path)
    stock = _stock(tmp_path)
    (stock / "sglang/__pycache__").mkdir()
    (stock / "sglang/__pycache__/activation.pyc").write_bytes(b"ambient")
    (stock / "sglang/.clang-format").write_text("ambient")
    stage = tmp_path / "native-stage"
    (stage / "cuda").mkdir(parents=True)
    (stage / "cuda/incumbent.bin").write_bytes(b"incumbent")
    identity = build_discovery_overlay_stage(
        inspect_discovery(bundle),
        stock_site_root=stock,
        native_stage_root=stage,
        policy=DEFAULT_DISCOVERY_POLICY,
        build_profile=_profile(),
    )
    envelope = stage / "dep_overlays/discovery"
    assert (envelope / "site/sglang/srt/layers/activation.py").read_text() == "VALUE = 2\n"
    assert (envelope / "site/sglang/untouched.py").read_text() == "UNCHANGED = True\n"
    assert not (envelope / "site/sglang/__pycache__").exists()
    assert not (envelope / "site/sglang/.clang-format").exists()
    assert tuple(row.path for row in identity.touched_files) == (
        "srt/layers/activation.py",
    )

    publication = publish_native_artifact(
        stage, tmp_path / "publications", build_spec_digest=_h("native-build")
    )
    reopened = reopen_discovery_overlay(
        publication, expected_identity_digest=identity.digest
    )
    assert reopened.root == publication.root / "dep_overlays/discovery"
    assert reopened.site_root == reopened.root / "site"
    assert reopened.identity == identity
    assert "cuda/incumbent.bin" in {row.path for row in publication.files}


def test_overlay_reopen_requires_identity_closed_envelope_and_read_only_mount(tmp_path):
    stage = tmp_path / "native-stage"
    stage.mkdir()
    identity = build_discovery_overlay_stage(
        inspect_discovery(_bundle(tmp_path)),
        stock_site_root=_stock(tmp_path),
        native_stage_root=stage,
        policy=DEFAULT_DISCOVERY_POLICY,
        build_profile=_profile(),
    )
    publication = publish_native_artifact(
        stage, tmp_path / "publications", build_spec_digest=_h("native-build")
    )
    with pytest.raises(DiscoveryError, match="identity digest mismatch"):
        reopen_discovery_overlay(
            publication, expected_identity_digest=_h("wrong")
        )
    with pytest.raises(DiscoveryError, match="not mounted read-only"):
        reopen_discovery_overlay(
            publication,
            expected_identity_digest=identity.digest,
            require_read_only=True,
            read_only_check=lambda _path: False,
        )
    assert reopen_discovery_overlay(
        publication,
        expected_identity_digest=identity.digest,
        require_read_only=True,
        read_only_check=lambda _path: True,
    ).identity == identity

    bad_stage = tmp_path / "bad-native-stage"
    bad_stage.mkdir()
    bad_identity = build_discovery_overlay_stage(
        inspect_discovery(_bundle(tmp_path, name="bad-bundle")),
        stock_site_root=_stock(tmp_path, name="bad-stock"),
        native_stage_root=bad_stage,
        policy=DEFAULT_DISCOVERY_POLICY,
        build_profile=_profile(),
    )
    (bad_stage / "dep_overlays/discovery/unexpected.txt").write_text("unexpected")
    bad_publication = publish_native_artifact(
        bad_stage,
        tmp_path / "bad-publications",
        build_spec_digest=_h("bad-native-build"),
    )
    with pytest.raises(DiscoveryError, match="envelope is not closed"):
        reopen_discovery_overlay(
            bad_publication, expected_identity_digest=bad_identity.digest
        )


def test_exact_context_mismatch_refuses_stage_build(tmp_path):
    bundle = _bundle(
        tmp_path,
        patch_text=_diff(
            "NOT PINNED\n",
            "VALUE = 2\n",
            "sglang/srt/layers/activation.py",
        ),
    )
    stage = tmp_path / "native-stage"
    stage.mkdir()
    with pytest.raises(DiscoveryError, match="context mismatch"):
        build_discovery_overlay_stage(
            inspect_discovery(bundle),
            stock_site_root=_stock(tmp_path),
            native_stage_root=stage,
            policy=DEFAULT_DISCOVERY_POLICY,
            build_profile=_profile(),
        )


def test_overlay_identity_binds_policy_profile_proposal_and_stock(tmp_path):
    stock = _stock(tmp_path)
    inspected = inspect_discovery(_bundle(tmp_path))
    first_stage = tmp_path / "first-stage"
    first_stage.mkdir()
    first = build_discovery_overlay_stage(
        inspected,
        stock_site_root=stock,
        native_stage_root=first_stage,
        policy=DEFAULT_DISCOVERY_POLICY,
        build_profile=_profile(),
    )
    (stock / "sglang/untouched.py").write_text("UNCHANGED = False\n")
    second_stage = tmp_path / "second-stage"
    second_stage.mkdir()
    second = build_discovery_overlay_stage(
        inspected,
        stock_site_root=stock,
        native_stage_root=second_stage,
        policy=DEFAULT_DISCOVERY_POLICY,
        build_profile=_profile(),
    )
    assert first.stock_tree_digest != second.stock_tree_digest
    assert first.digest != second.digest

    changed_policy = replace(
        DEFAULT_DISCOVERY_POLICY,
        forbidden_added_source=tuple(
            sorted((*DEFAULT_DISCOVERY_POLICY.forbidden_added_source, "evil.module"))
        ),
    )
    third_stage = tmp_path / "third-stage"
    third_stage.mkdir()
    third = build_discovery_overlay_stage(
        inspected,
        stock_site_root=stock,
        native_stage_root=third_stage,
        policy=changed_policy,
        build_profile=_profile(),
    )
    assert second.policy_digest != third.policy_digest

    assert replace(second, build_profile_digest=_h("another-profile")).digest != second.digest
    assert replace(second, proposal_digest=_h("another-proposal")).digest != second.digest


def test_ephemeral_arm_has_exact_bookends_and_no_permanent_stack_entry():
    incumbent = _incumbent()
    arm = DiscoveryArmPlan.create(
        incumbent=incumbent,
        incumbent_tree_digest=_h("incumbent-tree"),
        candidate_tree_digest=_h("candidate-tree"),
        proposal_digest=_h("proposal"),
        policy_digest=_h("policy"),
        build_profile_digest=_h("profile"),
        overlay_identity_digest=_h("overlay"),
    )
    expected_candidate = discovery_candidate_stack_digest(
        incumbent_stack_digest=incumbent.digest,
        incumbent_tree_digest=_h("incumbent-tree"),
        proposal_digest=_h("proposal"),
        policy_digest=_h("policy"),
        build_profile_digest=_h("profile"),
    )
    expected_delta = discovery_selected_delta_digest(
        proposal_digest=_h("proposal"),
        policy_digest=_h("policy"),
        build_profile_digest=_h("profile"),
    )
    assert arm.candidate_stack_digest == expected_candidate
    assert arm.selected_delta_digest == expected_delta
    assert arm.candidate_stack_digest != arm.incumbent_stack_digest
    assert arm.baseline_before == arm.baseline_after
    assert arm.baseline_before is not arm.baseline_after
    assert arm.challenger.stack_digest == arm.candidate_stack_digest
    assert arm.selected_delta_digest != arm.proposal_digest
    assert "contribution" not in json.dumps(arm.to_dict())

    with pytest.raises(DiscoveryError, match="ephemeral discovery identity"):
        replace(arm, candidate_stack_digest=_h("forged"))
    with pytest.raises(DiscoveryError, match="must differ"):
        replace(arm, candidate_tree_digest=arm.incumbent_tree_digest)

    another_result = replace(
        arm,
        overlay_identity_digest=_h("another-overlay"),
    )
    assert another_result.candidate_stack_digest == arm.candidate_stack_digest
    assert another_result.candidate_tree_digest == arm.candidate_tree_digest
    assert another_result.selected_delta_digest == arm.selected_delta_digest
    assert another_result.digest != arm.digest


def test_win_and_reviewed_promotion_are_bounded_value_records():
    win = DiscoveryWinRecord(
        arm_digest=_h("arm"),
        proposal_digest=_h("proposal"),
        overlay_identity_digest=_h("overlay"),
        qualification_digest=_h("qualification"),
        requested_promotion="new_singleton",
    )
    assert win.to_dict()["decision"] == "pass"
    promotion = DiscoveryPromotion(
        win_record_digest=win.digest,
        disposition="new_singleton",
        review_digest=_h("review"),
        subject="attention.new-boundary",
    )
    assert promotion.to_dict()["subject"] == "attention.new-boundary"
    assert "settlement" not in json.dumps(promotion.to_dict())

    bounty = DiscoveryPromotion(
        win_record_digest=win.digest,
        disposition="bounty_only",
        review_digest=_h("bounty-review"),
    )
    assert bounty.subject is None
    with pytest.raises(DiscoveryError, match="cannot name a shipping subject"):
        replace(bounty, subject="some.target")
    with pytest.raises(DiscoveryError, match="promotion subject"):
        DiscoveryPromotion(
            win_record_digest=win.digest,
            disposition="atomic_target",
            review_digest=_h("review-two"),
        )
