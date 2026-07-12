from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from optima.engine_tree import materialize_engine_tree
from optima.eval.engine_launch import (
    EngineLaunchError,
    EngineLaunchSpec,
    LogicalHardwareSpec,
    NativeBuildSpec,
    PhysicalHardwareBinding,
    TrustedLaunchBinding,
    native_compiler_policy_digest,
    native_patcher_digest,
    native_toolchain_digest,
    reopen_launch_tree,
    resolve_engine_launch,
    validate_native_build_spec,
    validate_runtime_preflight_receipt,
)
from optima.stack_manifest import EvaluationStackContext, EvaluationStackManifest
from optima.target_catalog import default_target_catalog


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _hardware(**changes: object) -> LogicalHardwareSpec:
    values: dict[str, object] = {
        "visible_gpu_count": 8,
        "architecture": "sm120",
        "topology_class": "pcie_switch",
        "topology_digest": _digest("topology"),
        "tp_size": 4,
        "ep_size": 1,
        "dp_size": 2,
        "device_policy_digest": _digest("device-policy"),
    }
    values.update(changes)
    return LogicalHardwareSpec(**values)  # type: ignore[arg-type]


def _native(tree_digest: str, **changes: object) -> NativeBuildSpec:
    values: dict[str, object] = {
        "tree_digest": tree_digest,
        "image_digest": _digest("image"),
        "platform_digest": _digest("platform"),
        "worker_distribution_digest": _digest("worker-dist"),
        "target_architecture": "sm120",
        "dependency_policy_digest": _digest("dependency-policy"),
    }
    values.update(changes)
    values.setdefault("toolchain_digest", native_toolchain_digest(
        image_digest=values["image_digest"],  # type: ignore[arg-type]
        platform_digest=values["platform_digest"],  # type: ignore[arg-type]
    ))
    values.setdefault("patcher_digest", native_patcher_digest(
        worker_distribution_digest=values["worker_distribution_digest"],  # type: ignore[arg-type]
    ))
    values.setdefault("compiler_flags_digest", native_compiler_policy_digest(
        image_digest=values["image_digest"],  # type: ignore[arg-type]
        worker_distribution_digest=values["worker_distribution_digest"],  # type: ignore[arg-type]
        dependency_policy_digest=values["dependency_policy_digest"],  # type: ignore[arg-type]
        target_architecture=values["target_architecture"],  # type: ignore[arg-type]
    ))
    return NativeBuildSpec(**values)  # type: ignore[arg-type]


def _launch(
    *,
    stack_digest: str,
    tree_digest: str,
    native: NativeBuildSpec | None = None,
    hardware: LogicalHardwareSpec | None = None,
    **changes: object,
) -> EngineLaunchSpec:
    build = native or _native(tree_digest)
    values: dict[str, object] = {
        "runtime_digest": _digest("runtime"),
        "base_engine_digest": _digest("base-engine"),
        "arena_digest": _digest("arena"),
        "stack_digest": stack_digest,
        "tree_digest": tree_digest,
        "image_digest": build.image_digest,
        "platform_digest": build.platform_digest,
        "controller_distribution_digest": _digest("controller-dist"),
        "worker_distribution_digest": build.worker_distribution_digest,
        "model_revision_digest": _digest("model-revision"),
        "model_manifest_digest": _digest("model-manifest"),
        "model_content_digest": _digest("model-content"),
        "validator_overlay_digest": _digest("validator-overlay"),
        "engine_config_digest": _digest("engine-config"),
        "seccomp_policy_digest": _digest("seccomp"),
        "resource_policy_digest": _digest("resources"),
        "native_build_spec_digest": build.digest,
        "hardware": hardware or _hardware(),
    }
    values.update(changes)
    return EngineLaunchSpec(**values)  # type: ignore[arg-type]


def _materialized_tree(tmp_path: Path):
    catalog = default_target_catalog()
    snapshot = catalog.snapshot()
    target_specs = {
        row["target_id"]: catalog.target_spec_digest(row["target_id"])
        for row in snapshot["targets"]
    }
    context = EvaluationStackContext(
        runtime_digest=_digest("tree-runtime"),
        base_engine_digest=_digest("tree-base"),
        arena_digest=_digest("tree-arena"),
        catalog_snapshot=snapshot,
        catalog_digest=catalog.digest,
        target_spec_digests=target_specs,
    )
    stack = EvaluationStackManifest(
        runtime_digest=context.runtime_digest,
        base_engine_digest=context.base_engine_digest,
        arena_digest=context.arena_digest,
        catalog_snapshot=snapshot,
        catalog_digest=catalog.digest,
        entries={},
    )
    return materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver={},
        destination=tmp_path / "materialized",
    )


def _physical(**changes: object) -> PhysicalHardwareBinding:
    values: dict[str, object] = {
        "physical_gpu_ids": tuple(f"GPU-{index}" for index in range(8)),
        "architecture": "sm120",
        "topology_class": "pcie_switch",
        "topology_digest": _digest("topology"),
        "tp_size": 4,
        "ep_size": 1,
        "dp_size": 2,
        "device_policy_digest": _digest("device-policy"),
    }
    values.update(changes)
    return PhysicalHardwareBinding(**values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class _Receipt:
    launch_identity: object


def _receipt(launch: EngineLaunchSpec, **changes: object) -> _Receipt:
    identity = dict(launch.runtime_preflight_identity)
    identity.update(changes)
    return _Receipt(identity)


def test_strict_canonical_round_trip_is_path_and_role_free() -> None:
    native = _native(_digest("tree"))
    launch = _launch(
        stack_digest=_digest("stack"), tree_digest=native.tree_digest, native=native
    )

    assert EngineLaunchSpec.from_dict(launch.to_dict()) == launch
    assert NativeBuildSpec.from_dict(native.to_dict()) == native
    assert LogicalHardwareSpec.from_dict(launch.hardware.to_dict()) == launch.hardware
    assert json.loads(launch.canonical_bytes) == launch.to_dict()
    assert json.loads(native.canonical_bytes) == native.to_dict()
    assert launch.digest == EngineLaunchSpec.from_dict(launch.to_dict()).digest
    assert set(launch.runtime_preflight_identity) == {
        "image_digest",
        "platform_digest",
        "worker_distribution_digest",
    }

    encoded = launch.canonical_bytes.decode()
    for forbidden in (
        "/tmp/engine",
        "GPU-0",
        "hotkey",
        "target_id",
        "score",
        "arm_role",
        "request_nonce",
    ):
        assert forbidden not in encoded


@pytest.mark.parametrize(
    "forbidden",
    (
        "bundle_path",
        "model_path",
        "physical_gpu_ids",
        "hotkey",
        "target_id",
        "score",
        "arm_role",
        "request_nonce",
    ),
)
def test_launch_schema_rejects_path_physical_and_economic_fields(
    forbidden: str,
) -> None:
    launch = _launch(stack_digest=_digest("stack"), tree_digest=_digest("tree"))
    row = launch.to_dict()
    row[forbidden] = "attacker-controlled"
    with pytest.raises(EngineLaunchError, match="schema mismatch"):
        EngineLaunchSpec.from_dict(row)


def test_nested_schemas_are_closed_and_versions_and_digests_are_strict() -> None:
    native = _native(_digest("tree"))
    launch = _launch(
        stack_digest=_digest("stack"), tree_digest=native.tree_digest, native=native
    )

    launch_row = copy.deepcopy(launch.to_dict())
    launch_row["hardware"]["gpu_path"] = "/dev/nvidia0"  # type: ignore[index]
    with pytest.raises(EngineLaunchError, match="logical hardware schema mismatch"):
        EngineLaunchSpec.from_dict(launch_row)

    native_row = native.to_dict()
    native_row["source_stem"] = "collision"
    with pytest.raises(EngineLaunchError, match="native build schema mismatch"):
        NativeBuildSpec.from_dict(native_row)

    for row, loader in (
        (launch.to_dict(), EngineLaunchSpec.from_dict),
        (native.to_dict(), NativeBuildSpec.from_dict),
        (launch.hardware.to_dict(), LogicalHardwareSpec.from_dict),
    ):
        changed = dict(row)
        changed["schema_version"] = 2
        with pytest.raises(EngineLaunchError, match="schema_version"):
            loader(changed)

    with pytest.raises(EngineLaunchError, match="all-zero"):
        replace(launch, runtime_digest="0" * 64)
    with pytest.raises(EngineLaunchError, match="canonical architecture"):
        replace(launch.hardware, architecture="SM_120")
    with pytest.raises(EngineLaunchError, match="positive integer"):
        replace(launch.hardware, visible_gpu_count=True)


def test_every_launch_identity_field_rotates_the_launch_digest() -> None:
    launch = _launch(stack_digest=_digest("stack"), tree_digest=_digest("tree"))

    digest_fields = (
        "runtime_digest",
        "base_engine_digest",
        "arena_digest",
        "stack_digest",
        "tree_digest",
        "image_digest",
        "platform_digest",
        "controller_distribution_digest",
        "worker_distribution_digest",
        "model_revision_digest",
        "model_manifest_digest",
        "model_content_digest",
        "validator_overlay_digest",
        "engine_config_digest",
        "seccomp_policy_digest",
        "resource_policy_digest",
        "native_build_spec_digest",
    )
    for field in digest_fields:
        changed = replace(launch, **{field: _digest(f"changed:{field}")})
        assert changed.digest != launch.digest, field

    hardware_changes = (
        {"visible_gpu_count": 16},
        {"architecture": "sm103"},
        {"topology_class": "nvlink_full"},
        {"topology_digest": _digest("changed topology")},
        {"tp_size": 8},
        {"ep_size": 2},
        {"dp_size": 1},
        {"device_policy_digest": _digest("changed device policy")},
    )
    for change in hardware_changes:
        changed = replace(launch, hardware=replace(launch.hardware, **change))
        assert changed.digest != launch.digest, change


def test_native_build_digest_binds_whole_tree_environment_and_policy() -> None:
    native = _native(_digest("tree"))
    variants = (
        _native(_digest("other tree")),
        _native(native.tree_digest, image_digest=_digest("other image")),
        _native(native.tree_digest, platform_digest=_digest("other platform")),
        _native(
            native.tree_digest,
            worker_distribution_digest=_digest("other worker"),
        ),
        _native(native.tree_digest, target_architecture="sm103"),
        _native(
            native.tree_digest,
            dependency_policy_digest=_digest("other dependencies"),
        ),
    )
    for variant in variants:
        assert variant.digest != native.digest

    for field in ("toolchain_digest", "patcher_digest", "compiler_flags_digest"):
        with pytest.raises(EngineLaunchError, match="grounded"):
            replace(native, **{field: _digest(f"unbound {field}")})

    launch = _launch(
        stack_digest=_digest("stack"), tree_digest=native.tree_digest, native=native
    )
    validate_native_build_spec(launch, native)

    other = _native(
        native.tree_digest,
        worker_distribution_digest=_digest("unregistered worker"),
    )
    with pytest.raises(EngineLaunchError, match="native_build_spec_digest"):
        validate_native_build_spec(launch, other)

    other_arch = _native(native.tree_digest, target_architecture="sm103")
    launch_with_other_digest = replace(
        launch, native_build_spec_digest=other_arch.digest
    )
    with pytest.raises(EngineLaunchError, match="target_architecture"):
        validate_native_build_spec(launch_with_other_digest, other_arch)


def test_resolution_reopens_tree_and_validates_all_local_bindings(
    tmp_path: Path,
) -> None:
    tree = _materialized_tree(tmp_path)
    native = _native(tree.tree_digest)
    launch = _launch(
        stack_digest=tree.stack_digest,
        tree_digest=tree.tree_digest,
        native=native,
    )
    binding = TrustedLaunchBinding(
        materialized_tree_root=tree.root,
        controller_distribution_digest=launch.controller_distribution_digest,
        native_build_spec=native,
        runtime_preflight_receipt=_receipt(launch),
        physical_hardware=_physical(),
    )

    resolved = resolve_engine_launch(launch, binding)

    assert resolved.spec is launch
    assert resolved.materialized_tree.tree_digest == tree.tree_digest
    assert resolved.materialized_tree.stack_digest == tree.stack_digest
    assert resolved.materialized_tree_root == tree.root.resolve()
    assert dict(resolved.runtime_preflight_identity) == dict(
        launch.runtime_preflight_identity
    )
    assert "GPU-0" not in launch.canonical_bytes.decode()
    assert str(tree.root) not in launch.canonical_bytes.decode()


def test_reopen_rejects_tree_digest_and_embedded_stack_split_brains(
    tmp_path: Path,
) -> None:
    tree = _materialized_tree(tmp_path)
    launch = _launch(stack_digest=tree.stack_digest, tree_digest=tree.tree_digest)

    with pytest.raises(EngineLaunchError, match="tree digest mismatch"):
        reopen_launch_tree(
            replace(launch, tree_digest=_digest("wrong tree")), tree.root
        )
    with pytest.raises(EngineLaunchError, match="stack_digest"):
        reopen_launch_tree(
            replace(launch, stack_digest=_digest("wrong stack")), tree.root
        )

    metadata = tree.root / "metadata" / "optima_engine_tree.json"
    metadata.chmod(0o644)
    with pytest.raises(EngineLaunchError, match="mode mismatch"):
        reopen_launch_tree(launch, tree.root)


def test_resolution_rejects_distribution_and_preflight_mismatches(
    tmp_path: Path,
) -> None:
    tree = _materialized_tree(tmp_path)
    native = _native(tree.tree_digest)
    launch = _launch(
        stack_digest=tree.stack_digest, tree_digest=tree.tree_digest, native=native
    )
    binding = TrustedLaunchBinding(
        materialized_tree_root=tree.root,
        controller_distribution_digest=launch.controller_distribution_digest,
        native_build_spec=native,
        runtime_preflight_receipt=_receipt(launch),
        physical_hardware=_physical(),
    )

    with pytest.raises(EngineLaunchError, match="controller distribution"):
        resolve_engine_launch(
            launch,
            replace(
                binding,
                controller_distribution_digest=_digest("wrong controller"),
            ),
        )
    with pytest.raises(EngineLaunchError, match="preflight launch_identity"):
        resolve_engine_launch(
            launch,
            replace(
                binding,
                runtime_preflight_receipt=_receipt(
                    launch, worker_distribution_digest=_digest("wrong worker")
                ),
            ),
        )

    extra = dict(launch.runtime_preflight_identity)
    extra["candidate_digest"] = _digest("candidate")
    with pytest.raises(EngineLaunchError, match="schema mismatch"):
        validate_runtime_preflight_receipt(launch, _Receipt(extra))


@pytest.mark.parametrize(
    "change,field",
    (
        ({"physical_gpu_ids": tuple(f"GPU-{i}" for i in range(7))}, "visible_gpu_count"),
        ({"architecture": "sm103"}, "architecture"),
        ({"topology_class": "nvlink_full"}, "topology_class"),
        ({"topology_digest": _digest("wrong topology")}, "topology_digest"),
        ({"tp_size": 8}, "tp_size"),
        ({"ep_size": 2}, "ep_size"),
        ({"dp_size": 1}, "dp_size"),
        ({"device_policy_digest": _digest("wrong policy")}, "device_policy_digest"),
    ),
)
def test_physical_gpu_realization_must_match_every_logical_field(
    change: dict[str, object], field: str
) -> None:
    with pytest.raises(EngineLaunchError, match=field):
        _physical(**change).validate_against(_hardware())


def test_physical_gpu_identifiers_are_unique_bounded_local_data() -> None:
    with pytest.raises(EngineLaunchError, match="duplicates"):
        _physical(physical_gpu_ids=("GPU-0", "GPU-0"))
    with pytest.raises(EngineLaunchError, match="invalid identifier"):
        _physical(physical_gpu_ids=("GPU-0,device=GPU-1",))
    with pytest.raises(EngineLaunchError, match="must not be empty"):
        _physical(physical_gpu_ids=())


def test_fresh_controller_import_graph_excludes_candidate_gpu_and_native_entry(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "candidate-imported"
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "candidate_probe.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
    )
    repository = Path(__file__).parents[1].resolve()
    script = f"""
import ctypes
import json
import sys
sys.path.insert(0, {str(repository)!r})
native_loads = []
ctypes.CDLL = lambda *args, **kwargs: native_loads.append([args, kwargs])
before = set(sys.modules)
import optima.eval.engine_launch
import optima.eval.runtime_preflight
import optima.eval.native_artifact
import optima.eval.oci_process
import optima.eval.oci_prebuild
new = sorted(set(sys.modules) - before)
forbidden = [
    name for name in new
    if name.split('.', 1)[0] in {{'torch', 'sglang', 'triton', 'flashinfer'}}
    or name.startswith('candidate_probe')
]
print(json.dumps({{'forbidden': forbidden, 'native_loads': native_loads}}))
"""
    environment = dict(os.environ)
    environment.update(
        {
            "OPTIMA_BUNDLE_PATH": str(candidate),
            "PYTHONNOUSERSITE": "1",
        }
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert json.loads(completed.stdout) == {
        "forbidden": [],
        "native_loads": [],
    }
    assert not marker.exists()
