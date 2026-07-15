from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from optima.discovery import (
    DEFAULT_DISCOVERY_POLICY,
    DISCOVERY_ABI_VERSION,
    DiscoveryBuildProfile,
    build_discovery_overlay_stage,
    inspect_discovery,
    reopen_discovery_engine_binding,
    reopen_discovery_overlay,
)
from optima.engine_tree import (
    materialize_discovery_engine_tree,
    materialize_engine_tree,
)
from optima.eval.engine_launch import (
    EngineLaunchSpec,
    LogicalHardwareSpec,
    NativeBuildSpec,
    PhysicalHardwareBinding,
    TrustedLaunchBinding,
    native_compiler_policy_digest,
    native_patcher_digest,
    native_toolchain_digest,
    resolve_engine_launch,
)
from optima.eval.oci_prebuild import (
    OCIPrebuildConfig,
    OCIPrebuildError,
    OCIPrebuildPolicy,
    PREBUILD_RECEIPT,
    PREBUILD_SCHEMA,
    _write_compile_profile,
    build_prebuild_argv,
    container_build,
    run_oci_prebuild,
)
from optima.eval.native_compile_profile import NativeCuTeCompileProfile
from optima.eval.native_artifact import publish_native_artifact
from optima.eval.oci_process import (
    CommandResult,
    OCIAttachedDiagnostic,
    OCIProcessManager,
    OCIProcessResult,
)
from optima.eval.runtime_preflight import RuntimePreflightReceipt
from optima.stack_identity import canonical_json_bytes
from optima.stack_manifest import EvaluationStackContext, EvaluationStackManifest
from optima.target_catalog import default_target_catalog


DOCKER = "/usr/bin/docker"
IMAGE_ID = "sha256:" + "a" * 64


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _tree(tmp_path: Path):
    catalog = default_target_catalog()
    snapshot = catalog.snapshot()
    context = EvaluationStackContext(
        runtime_digest=_digest("tree-runtime"),
        base_engine_digest=_digest("tree-base"),
        arena_digest=_digest("tree-arena"),
        catalog_snapshot=snapshot,
        catalog_digest=catalog.digest,
        target_spec_digests={
            row["target_id"]: catalog.target_spec_digest(row["target_id"])
            for row in snapshot["targets"]
        },
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
        destination=tmp_path / "tree",
    )


def _discovery_source(root: Path) -> Path:
    (root / "patches").mkdir(parents=True)
    (root / "manifest.toml").write_text(
        'bundle_id = "prebuild-discovery"\n'
        f'abi_version = "{DISCOVERY_ABI_VERSION}"\n'
        'build_profile = "prebuild-sm120-tp8"\n'
        'patches = ["patches/change.patch"]\n'
        'dependencies = ["cuda13"]\n'
        'conflicts = []\n'
        'requested_promotion = "new_singleton"\n'
        "\n[applicability]\n"
        'arenas = ["minimax-m3-rtx-tp8-v1"]\n'
        'models = ["minimax-m3-nvfp4"]\n'
        'architectures = ["sm120"]\n'
        "tensor_parallel_sizes = [8]\n"
    )
    (root / "patches" / "change.patch").write_text(
        "--- a/sglang/srt/layers/activation.py\n"
        "+++ b/sglang/srt/layers/activation.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )
    return root


def _discovery_profile() -> DiscoveryBuildProfile:
    return DiscoveryBuildProfile(
        profile_id="prebuild-sm120-tp8",
        sglang_version=DEFAULT_DISCOVERY_POLICY.sglang_version,
        arena="minimax-m3-rtx-tp8-v1",
        model="minimax-m3-nvfp4",
        architecture="sm120",
        tensor_parallel_size=8,
        features=("cuda13",),
        build_inputs=(("image", _digest("discovery-image")),),
    )


def _discovery_tree(tmp_path: Path):
    incumbent_root = tmp_path / "incumbent"
    incumbent_root.mkdir()
    incumbent = _tree(incumbent_root)
    discovery = inspect_discovery(_discovery_source(tmp_path / "proposal"))
    return materialize_discovery_engine_tree(
        incumbent.root,
        discovery,
        policy=DEFAULT_DISCOVERY_POLICY,
        build_profile=_discovery_profile(),
        destination=tmp_path / "discovery-tree",
    )


def _stock_sglang(tmp_path: Path) -> Path:
    site = tmp_path / "stock-site"
    package = site / "sglang"
    (package / "srt" / "layers").mkdir(parents=True)
    (package / "__init__.py").write_text('__version__ = "stock"\n')
    (package / "srt" / "layers" / "activation.py").write_text("VALUE = 1\n")
    return site


def _policy(**changes: object) -> OCIPrebuildPolicy:
    values: dict[str, object] = {
        "uid": max(1, os.getuid()),
        "gid": max(1, os.getgid()),
        "cpu_millis": 8_000,
        "memory_bytes": 32 << 30,
        "pids_limit": 4_096,
        "tmpfs_bytes": 512 << 20,
        "stage_bytes": 16 << 30,
        "stage_inodes": 100_000,
        "timeout_seconds": 7_200,
        "native_compile_timeout_seconds": 6_000,
        "container_python": "/usr/local/bin/python3",
        "build_path": (
            "/usr/local/cuda/bin",
            "/usr/local/bin",
            "/usr/bin",
            "/bin",
        ),
        "build_tmpdir": "/tmp",
        "pinned_build_roots": (
            "/usr/include",
            "/usr/lib",
            "/usr/local/cuda",
            "/usr/local/include",
            "/usr/local/lib/python3.12/dist-packages",
        ),
        "runtime_policy_digest": _digest("runtime-policy"),
    }
    values.update(changes)
    return OCIPrebuildPolicy(**values)  # type: ignore[arg-type]


def _hardware() -> LogicalHardwareSpec:
    return LogicalHardwareSpec(
        visible_gpu_count=8,
        architecture="sm120",
        topology_class="pcie_switch",
        topology_digest=_digest("topology"),
        tp_size=8,
        ep_size=1,
        dp_size=1,
        device_policy_digest=_digest("device-policy"),
    )


def _physical() -> PhysicalHardwareBinding:
    return PhysicalHardwareBinding(
        physical_gpu_ids=tuple(f"GPU-{index}" for index in range(8)),
        architecture="sm120",
        topology_class="pcie_switch",
        topology_digest=_digest("topology"),
        tp_size=8,
        ep_size=1,
        dp_size=1,
        device_policy_digest=_digest("device-policy"),
    )


def _preflight(
    *,
    image: str,
    platform: str,
    worker: str,
    policy: OCIPrebuildPolicy,
    sglang_version: str = "0.0.0.dev1",
):
    return RuntimePreflightReceipt(
        schema="optima-runtime-preflight-v2",
        requested_image="registry.example/optima@sha256:" + image,
        image_digest=image,
        local_image_id=IMAGE_ID,
        repo_digests=("registry.example/optima@sha256:" + image,),
        oci_platform="linux/amd64",
        platform_digest=platform,
        docker_binary=DOCKER,
        uid=policy.uid,
        gid=policy.gid,
        sglang_version=sglang_version,
        worker_distribution="optima-harness",
        worker_version="0.0.1",
        worker_distribution_digest=worker,
        worker_file_count=200,
        worker_total_bytes=1_000_000,
        python_implementation="cpython",
        python_executable="/usr/local/bin/python3",
        python_version="3.12.0",
        python_abi="cpython-312-x86_64-linux-gnu",
        python_platform="linux-x86_64",
        machine="x86_64",
        package_versions=(),
        cudart_library="libcudart.so.13",
        cuda_visible_devices="",
        nvidia_visible_devices="void",
        security_argv_sha256=_digest("preflight-argv"),
    )


def _case(
    tmp_path: Path,
    *,
    policy: OCIPrebuildPolicy | None = None,
    tree=None,
    sglang_version: str = "0.0.0.dev1",
):
    tree = tree or _tree(tmp_path)
    policy = policy or _policy()
    seccomp = tmp_path / "seccomp.json"
    seccomp.write_text('{"defaultAction":"SCMP_ACT_ERRNO"}\n')
    image = _digest("image")
    platform = _digest("platform")
    worker = _digest("worker")
    native = NativeBuildSpec(
        tree_digest=tree.tree_digest,
        image_digest=image,
        platform_digest=platform,
        worker_distribution_digest=worker,
        toolchain_digest=native_toolchain_digest(
            image_digest=image, platform_digest=platform
        ),
        patcher_digest=native_patcher_digest(
            worker_distribution_digest=worker
        ),
        compiler_flags_digest=native_compiler_policy_digest(
            image_digest=image,
            worker_distribution_digest=worker,
            dependency_policy_digest=policy.dependency_policy_digest,
            target_architecture="sm120",
        ),
        target_architecture="sm120",
        dependency_policy_digest=policy.dependency_policy_digest,
    )
    launch = EngineLaunchSpec(
        runtime_digest=_digest("runtime"),
        base_engine_digest=_digest("engine"),
        arena_digest=_digest("arena"),
        stack_digest=tree.stack_digest,
        tree_digest=tree.tree_digest,
        image_digest=image,
        platform_digest=platform,
        controller_distribution_digest=_digest("controller"),
        worker_distribution_digest=worker,
        model_revision_digest=_digest("model-revision"),
        model_manifest_digest=_digest("model-manifest"),
        model_content_digest=_digest("model-content"),
        validator_overlay_digest=_digest("validator-overlay"),
        engine_config_digest=_digest("engine-config"),
        seccomp_policy_digest=hashlib.sha256(seccomp.read_bytes()).hexdigest(),
        resource_policy_digest=policy.resource_policy_digest,
        native_build_spec_digest=native.digest,
        hardware=_hardware(),
    )
    preflight = _preflight(
        image=image,
        platform=platform,
        worker=worker,
        policy=policy,
        sglang_version=sglang_version,
    )
    binding = TrustedLaunchBinding(
        materialized_tree_root=tree.root,
        controller_distribution_digest=launch.controller_distribution_digest,
        native_build_spec=native,
        runtime_preflight_receipt=preflight,
        physical_hardware=_physical(),
    )
    config = OCIPrebuildConfig(
        docker_binary=DOCKER,
        recovery_root=(tmp_path / "recovery").absolute(),
        publication_root=(tmp_path / "publications").absolute(),
        seccomp_profile=seccomp.absolute(),
        executor_id="validator-a",
        policy=policy,
    )
    return tree, launch, binding, preflight, config


def _write_receipt(
    stage: Path,
    *,
    launch: EngineLaunchSpec,
    native: NativeBuildSpec,
    discovery_overlay_identity_digest: str | None = None,
) -> None:
    entries = sorted((*[path.name for path in stage.iterdir()], PREBUILD_RECEIPT))
    row = {
        "build_spec_digest": native.digest,
        "rebuild_applied": False,
        "schema": PREBUILD_SCHEMA,
        "stage_entries": entries,
        "target_architecture": native.target_architecture,
        "tree_digest": launch.tree_digest,
    }
    if discovery_overlay_identity_digest is not None:
        row["discovery_overlay_identity_digest"] = discovery_overlay_identity_digest
    (stage / PREBUILD_RECEIPT).write_bytes(canonical_json_bytes(row) + b"\n")


def _native_with_dependency(native: NativeBuildSpec, dependency: str) -> NativeBuildSpec:
    return replace(
        native,
        dependency_policy_digest=dependency,
        compiler_flags_digest=native_compiler_policy_digest(
            image_digest=native.image_digest,
            worker_distribution_digest=native.worker_distribution_digest,
            dependency_policy_digest=dependency,
            target_architecture=native.target_architecture,
        ),
    )


class _Controls:
    def __call__(self, argv, *, timeout_s, max_output_bytes):
        row = tuple(argv)
        if row[1:3] == ("container", "ls"):
            return CommandResult(0, b"", b"")
        return CommandResult(0, b"", b"")


def _compile_profile() -> NativeCuTeCompileProfile:
    return NativeCuTeCompileProfile(
        logical_architecture="sm103",
        compiler_architecture="sm_103a",
        image_digest=_digest("profile-image"),
        platform_digest=_digest("profile-platform"),
        worker_distribution_digest=_digest("profile-worker"),
        logical_hardware_digest=_digest("profile-hardware"),
        device_policy_digest=_digest("profile-device"),
        topology_digest=_digest("profile-topology"),
        visible_gpu_count=8,
        tp_size=4,
        ep_size=1,
        dp_size=2,
        constants={"max_active_clusters.cluster_size_1": 148},
        measurement_digest=_digest("profile-measurement"),
    )


def test_compile_profile_staging_overrides_restrictive_controller_umask(
    tmp_path: Path,
) -> None:
    profile = _compile_profile()
    destination = tmp_path / "compile-profile.json"
    previous = os.umask(0o077)
    try:
        _write_compile_profile(profile, destination)
    finally:
        os.umask(previous)
    assert destination.stat().st_mode & 0o777 == 0o444
    assert destination.read_bytes() == profile.canonical_bytes


def test_policy_binds_resource_and_native_dependency_inputs(tmp_path: Path) -> None:
    policy = _policy()
    resource_changes = {
        "uid": policy.uid + 1,
        "cpu_millis": 9_000,
        "memory_bytes": 33 << 30,
        "stage_bytes": 17 << 30,
        "timeout_seconds": 7_201,
        "container_python": "/usr/bin/python3",
        "runtime_policy_digest": _digest("other runtime"),
    }
    for field, value in resource_changes.items():
        assert (
            replace(policy, **{field: value}).resource_policy_digest
            != policy.resource_policy_digest
        )
    assert (
        replace(
            policy,
            cpuset_cpus="0-3,8-11",
            cpuset_mems="0",
        ).resource_policy_digest
        != policy.resource_policy_digest
    )
    dependency_changes = {
        "build_path": ("/usr/bin", "/bin"),
        "build_tmpdir": "/var/tmp",
        "container_python": "/usr/bin/python3",
        "native_compile_timeout_seconds": 5_999,
        "pinned_build_roots": ("/usr/include", "/usr/lib"),
    }
    for field, value in dependency_changes.items():
        assert (
            replace(policy, **{field: value}).dependency_policy_digest
            != policy.dependency_policy_digest
        )

    _tree_row, _launch, binding, _preflight, _config_row = _case(tmp_path)
    changed_policy = replace(policy, container_python="/usr/bin/python3")
    changed_native = _native_with_dependency(
        binding.native_build_spec, changed_policy.dependency_policy_digest
    )
    assert changed_native.digest != binding.native_build_spec.digest


def test_prebuild_cpuset_policy_rejects_partial_or_noncanonical_sets() -> None:
    policy = _policy()
    for cpus, mems in (
        ("0-7", None),
        (None, "0"),
        ("0,1", "0"),
        ("0-7", "00"),
        ("0-6", "0"),
    ):
        with pytest.raises(OCIPrebuildError, match="cpuset|cpu_millis"):
            replace(policy, cpuset_cpus=cpus, cpuset_mems=mems)


def test_exact_prebuild_argv_has_only_two_mounts_no_gpu_no_egress_no_caps(
    tmp_path: Path,
) -> None:
    isolated = _policy(cpuset_cpus="0-3,8-11", cpuset_mems="0")
    tree, launch, binding, preflight, config = _case(tmp_path, policy=isolated)
    resolved = resolve_engine_launch(launch, binding)
    manager = OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=config.recovery_root,
        executor_id=config.executor_id,
        runner=_Controls(),
    )
    lease = manager.register(
        lease_id="prebuild-test",
        container_name="optima-prebuild-test",
        mount_relpaths=("stage",),
        stage_relpaths=("seccomp.json",),
    )
    argv = build_prebuild_argv(
        lease=lease,
        resolved=resolved,
        preflight=preflight,
        config=config,
        stage_path=lease.mount_paths[0],
        seccomp_path=lease.stage_paths[0],
    )

    assert argv[: len(lease.run_prefix(DOCKER))] == lease.run_prefix(DOCKER)
    assert "--network=none" in argv and "--read-only" in argv
    assert "--ipc=none" in argv and not any(value.startswith("--pid=") for value in argv)
    assert argv.count("--cap-drop=ALL") == 1
    assert not any(value.startswith("--cap-add") for value in argv)
    assert "--security-opt=no-new-privileges=true" in argv
    assert f"--security-opt=seccomp={lease.stage_paths[0]}" in argv
    assert "--cpuset-cpus=0-3,8-11" in argv
    assert "--cpuset-mems=0" in argv
    mounts = [value for value in argv if value.startswith("--mount=")]
    assert len(mounts) == 2
    assert str(tree.root) in mounts[0] and "readonly" in mounts[0]
    assert str(lease.mount_paths[0]) in mounts[1] and "readonly" not in mounts[1]
    assert not any(
        "/models" in value or "/root" in value or "docker.sock" in value
        for value in argv
    )
    assert not any("--gpus" in value or "--device" in value for value in argv)
    env_rows = [value for value in argv if value.startswith("--env=")]
    assert any(
        "OPTIMA_NATIVE_BUILD_SPEC_DIGEST=" + binding.native_build_spec.digest in value
        for value in env_rows
    )
    assert any("OPTIMA_BUILD_PATH=" in value for value in env_rows)
    assert any("OPTIMA_BUILD_TMPDIR=/tmp" in value for value in env_rows)
    assert any("OPTIMA_NATIVE_COMPILE_TIMEOUT_S=6000" in value for value in env_rows)
    env_keys = {value.split("=", 2)[1] for value in env_rows}
    assert not any(
        key.upper().endswith("PROXY") or key.startswith("LD_") or key == "PYTHONPATH"
        for key in env_keys
    )
    assert argv[-5:] == (
        IMAGE_ID,
        "-I",
        "-m",
        "optima.eval.oci_prebuild",
        "--container-build",
    )


def test_profiled_prebuild_adds_one_read_only_profile_mount_and_digest_env(
    tmp_path: Path,
) -> None:
    tree, launch, binding, preflight, config = _case(tmp_path)
    resolved = resolve_engine_launch(launch, binding)
    profile_digest = _digest("cute-profile")
    profiled = replace(
        resolved,
        native_compile_profile=SimpleNamespace(digest=profile_digest),
    )
    manager = OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=config.recovery_root,
        executor_id=config.executor_id,
        runner=_Controls(),
    )
    lease = manager.register(
        lease_id="profiled-prebuild-test",
        container_name="optima-profiled-prebuild-test",
        mount_relpaths=("stage",),
        stage_relpaths=("seccomp.json", "cute-profile.json"),
    )
    argv = build_prebuild_argv(
        lease=lease,
        resolved=profiled,
        preflight=preflight,
        config=config,
        stage_path=lease.mount_paths[0],
        seccomp_path=lease.stage_paths[0],
        compile_profile_path=lease.stage_paths[1],
    )
    profile_mounts = [
        value
        for value in argv
        if value.startswith("--mount=") and "cute-compile-profile.json" in value
    ]
    assert len(profile_mounts) == 1
    assert "readonly" in profile_mounts[0]
    assert f"--env=OPTIMA_CUTE_COMPILE_PROFILE_DIGEST={profile_digest}" in argv
    assert "--env=OPTIMA_CUTE_COMPILE_PROFILE=/optima/cute-compile-profile.json" in argv

    with pytest.raises(OCIPrebuildError, match="does not match launch authority"):
        build_prebuild_argv(
            lease=lease,
            resolved=profiled,
            preflight=preflight,
            config=config,
            stage_path=lease.mount_paths[0],
            seccomp_path=lease.stage_paths[0],
        )


@pytest.mark.skipif(sys.platform != "linux", reason="production publication uses Linux renameat2")
def test_run_builds_publishes_reopens_and_then_reuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    manager = OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=config.recovery_root,
        executor_id=config.executor_id,
        runner=_Controls(),
    )
    stage_holder: list[Path] = []

    def mount(_lease, path, **_kwargs):
        Path(path).mkdir(parents=True)
        stage_holder.append(Path(path))
        return Path(path)

    def run(_lease, _argv, *, timeout_s, stdin_bytes=b""):
        assert timeout_s == config.policy.timeout_seconds and stdin_bytes == b""
        _write_receipt(stage_holder[-1], launch=launch, native=binding.native_build_spec)
        return OCIProcessResult(0, 1.25)

    monkeypatch.setattr(manager, "mount_tmpfs", mount)
    monkeypatch.setattr(manager, "run", run)
    first = run_oci_prebuild(launch, binding, config, manager=manager)
    assert first.container_elapsed_seconds == 1.25
    assert first.publication.root.is_dir()
    assert first.publication.build_spec_digest == binding.native_build_spec.digest
    assert first.publication.root.stat().st_mode & 0o777 == 0o555
    assert first.discovery_overlay_identity_digest is None
    assert not manager.leases_root.joinpath("prebuild-test.json").exists()

    second = run_oci_prebuild(launch, binding, config, manager=manager)
    assert second.reused and second.container_elapsed_seconds is None
    assert second.discovery_overlay_identity_digest is None
    assert second.publication.publication_digest == first.publication.publication_digest


@pytest.mark.skipif(sys.platform != "linux", reason="production publication uses Linux renameat2")
def test_discovery_run_reopens_publication_and_reuses_bound_overlay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = _discovery_tree(tmp_path)
    stock_site = _stock_sglang(tmp_path)
    policy = _policy(pinned_build_roots=(str(stock_site.resolve()),))
    _tree_row, launch, binding, _preflight_row, config = _case(
        tmp_path,
        policy=policy,
        tree=tree,
        sglang_version=DEFAULT_DISCOVERY_POLICY.sglang_version,
    )
    discovery = reopen_discovery_engine_binding(tree)
    manager = OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=config.recovery_root,
        executor_id=config.executor_id,
        runner=_Controls(),
    )
    stage_holder: list[Path] = []

    def mount(_lease, path, **_kwargs):
        Path(path).mkdir(parents=True)
        stage_holder.append(Path(path))
        return Path(path)

    def run(_lease, _argv, *, timeout_s, stdin_bytes=b""):
        assert timeout_s == config.policy.timeout_seconds and stdin_bytes == b""
        identity = build_discovery_overlay_stage(
            discovery.discovery,
            stock_site_root=stock_site,
            native_stage_root=stage_holder[-1],
            policy=discovery.policy,
            build_profile=discovery.build_profile,
        )
        _write_receipt(
            stage_holder[-1],
            launch=launch,
            native=binding.native_build_spec,
            discovery_overlay_identity_digest=identity.digest,
        )
        return OCIProcessResult(0, 1.5)

    monkeypatch.setattr(manager, "mount_tmpfs", mount)
    monkeypatch.setattr(manager, "run", run)
    first = run_oci_prebuild(launch, binding, config, manager=manager)
    assert first.discovery_overlay_identity_digest is not None
    reopened = reopen_discovery_overlay(
        first.publication,
        expected_identity_digest=first.discovery_overlay_identity_digest,
    )
    assert reopened.identity.proposal_digest == discovery.discovery.proposal_digest

    second = run_oci_prebuild(launch, binding, config, manager=manager)
    assert second.reused
    assert second.discovery_overlay_identity_digest == first.discovery_overlay_identity_digest
    assert second.publication.publication_digest == first.publication.publication_digest


def test_cached_discovery_receipt_cannot_lie_about_published_overlay(
    tmp_path: Path,
) -> None:
    tree = _discovery_tree(tmp_path)
    stock_site = _stock_sglang(tmp_path)
    policy = _policy(pinned_build_roots=(str(stock_site.resolve()),))
    _tree_row, launch, binding, _preflight_row, config = _case(
        tmp_path,
        policy=policy,
        tree=tree,
        sglang_version=DEFAULT_DISCOVERY_POLICY.sglang_version,
    )
    discovery = reopen_discovery_engine_binding(tree)
    stage = tmp_path / "forged-stage"
    stage.mkdir()
    build_discovery_overlay_stage(
        discovery.discovery,
        stock_site_root=stock_site,
        native_stage_root=stage,
        policy=discovery.policy,
        build_profile=discovery.build_profile,
    )
    _write_receipt(
        stage,
        launch=launch,
        native=binding.native_build_spec,
        discovery_overlay_identity_digest=_digest("wrong-overlay-identity"),
    )
    publish_native_artifact(
        stage,
        config.publication_root,
        build_spec_digest=binding.native_build_spec.digest,
    )

    with pytest.raises(OCIPrebuildError, match="overlay publication cannot reopen"):
        run_oci_prebuild(launch, binding, config)


@pytest.mark.parametrize("deadline", (float("nan"), float("inf"), -float("inf"), True, "10"))
def test_prebuild_rejects_nonfinite_or_non_numeric_deadline_before_work(
    tmp_path: Path, deadline
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    with pytest.raises(OCIPrebuildError, match="deadline must be a finite"):
        run_oci_prebuild(launch, binding, config, deadline=deadline)
    assert not config.recovery_root.exists()


def test_prebuild_rejects_expired_deadline_before_binding_or_lease(
    tmp_path: Path,
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    manager = OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=config.recovery_root,
        executor_id=config.executor_id,
        runner=_Controls(),
        clock=lambda: 10.0,
    )
    with pytest.raises(OCIPrebuildError, match="deadline expired during binding"):
        run_oci_prebuild(launch, binding, config, manager=manager, deadline=10.0)
    assert list(manager.leases_root.iterdir()) == []


def test_prebuild_rechecks_deadline_after_binding_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import optima.eval.oci_prebuild as prebuild_mod

    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    now = {"value": 100.0}
    manager = OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=config.recovery_root,
        executor_id=config.executor_id,
        runner=_Controls(),
        clock=lambda: now["value"],
    )
    real_validate = prebuild_mod._validate_binding

    def validate(*args, **kwargs):
        result = real_validate(*args, **kwargs)
        now["value"] = 106.0
        return result

    monkeypatch.setattr(prebuild_mod, "_validate_binding", validate)
    with pytest.raises(OCIPrebuildError, match="deadline expired during binding"):
        run_oci_prebuild(launch, binding, config, manager=manager, deadline=105.0)
    assert list(manager.leases_root.iterdir()) == []


@pytest.mark.parametrize(
    "deadline,expected_timeout",
    ((105.0, 5.0), (10_000.0, 7_200.0)),
)
def test_prebuild_container_timeout_is_capped_by_absolute_deadline_and_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    deadline: float,
    expected_timeout: float,
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    manager = OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=config.recovery_root,
        executor_id=config.executor_id,
        runner=_Controls(),
        clock=lambda: 100.0,
    )
    observed = []

    def mount(_lease, path, **_kwargs):
        Path(path).mkdir(parents=True)
        return Path(path)

    def run(_lease, _argv, *, timeout_s, stdin_bytes=b""):
        observed.append((timeout_s, stdin_bytes))
        return OCIProcessResult(9, 0.1)

    monkeypatch.setattr(manager, "mount_tmpfs", mount)
    monkeypatch.setattr(manager, "run", run)
    with pytest.raises(OCIPrebuildError, match="container exited 9"):
        run_oci_prebuild(
            launch, binding, config, manager=manager, deadline=deadline
        )
    assert observed == [(expected_timeout, b"")]
    assert list(manager.leases_root.iterdir()) == []


def test_prebuild_failure_preserves_only_bounded_terminal_safe_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    manager = OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=config.recovery_root,
        executor_id=config.executor_id,
        runner=_Controls(),
    )

    def mount(_lease, path, **_kwargs):
        Path(path).mkdir(parents=True)
        return Path(path)

    tail = b"compile failed: \x1b[31mTRACEBACK-TAIL\x1b[0m"
    diagnostic = OCIAttachedDiagnostic(tail, True, True, client_returncode=9)

    def run(_lease, _argv, *, timeout_s, stdin_bytes=b""):
        assert timeout_s == config.policy.timeout_seconds and stdin_bytes == b""
        return OCIProcessResult(9, 0.1, diagnostic)

    monkeypatch.setattr(manager, "mount_tmpfs", mount)
    monkeypatch.setattr(manager, "run", run)
    with pytest.raises(OCIPrebuildError) as caught:
        run_oci_prebuild(launch, binding, config, manager=manager)
    rendered = str(caught.value)
    assert "container exited 9" in rendered
    assert diagnostic.stderr_sha256 in rendered
    assert "TRACEBACK-TAIL" in rendered
    assert "\x1b" not in rendered and "\\x1b" in rendered
    assert list(manager.leases_root.iterdir()) == []


def test_prebuild_expiry_after_container_prevents_publication_and_releases_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    manager = OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=config.recovery_root,
        executor_id=config.executor_id,
        runner=_Controls(),
    )
    now = {"value": 100.0}
    monkeypatch.setattr(manager, "clock", lambda: now["value"])
    stage_holder: list[Path] = []

    def mount(_lease, path, **_kwargs):
        Path(path).mkdir(parents=True)
        stage_holder.append(Path(path))
        return Path(path)

    def run(_lease, _argv, *, timeout_s, stdin_bytes=b""):
        assert timeout_s == 5.0 and stdin_bytes == b""
        _write_receipt(
            stage_holder[-1], launch=launch, native=binding.native_build_spec
        )
        now["value"] = 106.0
        return OCIProcessResult(0, 1.0)

    monkeypatch.setattr(manager, "mount_tmpfs", mount)
    monkeypatch.setattr(manager, "run", run)
    with pytest.raises(OCIPrebuildError, match="deadline expired during container"):
        run_oci_prebuild(
            launch, binding, config, manager=manager, deadline=105.0
        )
    assert not (config.publication_root / binding.native_build_spec.digest[:2]).exists()
    assert list(manager.leases_root.iterdir()) == []


def test_prebuild_deadline_fails_closed_on_nonfinite_manager_clock(
    tmp_path: Path,
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    manager = OCIProcessManager(
        docker_binary=DOCKER,
        recovery_root=config.recovery_root,
        executor_id=config.executor_id,
        runner=_Controls(),
        clock=lambda: float("nan"),
    )
    with pytest.raises(OCIPrebuildError, match="clock returned a non-finite"):
        run_oci_prebuild(launch, binding, config, manager=manager, deadline=105.0)


@pytest.mark.parametrize(
    "mutator,match",
    (
        (lambda launch, binding, config: (replace(launch, resource_policy_digest=_digest("bad")), binding, config), "resource policy"),
        (lambda launch, binding, config: (launch, replace(binding, native_build_spec=_native_with_dependency(binding.native_build_spec, _digest("bad"))), config), "native_build_spec_digest"),
        (lambda launch, binding, config: (launch, binding, replace(config, docker_binary="/opt/docker")), "Docker clients differ"),
    ),
)
def test_binding_mismatch_rejects_before_lease(
    tmp_path: Path, mutator, match: str
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    launch, binding, config = mutator(launch, binding, config)
    with pytest.raises((OCIPrebuildError, ValueError), match=match):
        run_oci_prebuild(launch, binding, config)
    assert not config.recovery_root.exists()


def test_discovery_profile_must_match_image_preflight_before_lease(
    tmp_path: Path,
) -> None:
    tree = _discovery_tree(tmp_path)
    _tree_row, launch, binding, _preflight_row, config = _case(
        tmp_path,
        tree=tree,
        sglang_version="wrong-image-sglang",
    )

    with pytest.raises(OCIPrebuildError, match="image SGLang preflight"):
        run_oci_prebuild(launch, binding, config)

    assert not config.recovery_root.exists()


def test_publication_and_recovery_roots_must_not_overlap_materialized_tree_or_each_other(
    tmp_path: Path,
) -> None:
    tree, launch, binding, _preflight, config = _case(tmp_path)
    with pytest.raises(OCIPrebuildError, match="must not overlap"):
        run_oci_prebuild(
            launch,
            binding,
            replace(config, publication_root=tree.root / "published"),
        )
    with pytest.raises(OCIPrebuildError, match="must not overlap"):
        run_oci_prebuild(
            launch,
            binding,
            replace(config, publication_root=config.recovery_root / "published"),
        )


def test_container_build_scrubs_ambient_environment_and_applies_build_phase(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import optima.eval.oci_prebuild as prebuild_mod
    import optima.rebuild as rebuild_mod

    tree = _tree(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    monkeypatch.setattr(prebuild_mod, "CONTAINER_TREE", str(tree.root))
    monkeypatch.setattr(prebuild_mod, "CONTAINER_STAGE", str(stage))
    seen = []
    monkeypatch.setattr(
        rebuild_mod,
        "apply_rebuild_plan",
        lambda path, *, phase: seen.append((Path(path), phase)) or False,
    )
    original_environment = dict(os.environ)
    required = {
        "OPTIMA_NATIVE_BUILD_SPEC_DIGEST": _digest("build"),
        "OPTIMA_ENGINE_TREE_DIGEST": tree.tree_digest,
        "OPTIMA_TARGET_GPU_ARCH": "sm120",
        "OPTIMA_NATIVE_ARTIFACT_STAGE": str(stage),
        "OPTIMA_PINNED_BUILD_ROOTS": "/usr/include:/usr/lib",
        "OPTIMA_BUILD_PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "OPTIMA_BUILD_TMPDIR": "/tmp",
        "OPTIMA_NATIVE_COMPILE_TIMEOUT_S": "60",
        "OPTIMA_REBUILD_CONTAINER": "1",
        "HTTPS_PROXY": "https://must-not-survive.invalid",
        "LD_PRELOAD": "/tmp/evil.so",
        "PYTHONPATH": "/tmp/evil",
    }
    try:
        os.environ.update(required)
        receipt = container_build()
        assert receipt == stage / PREBUILD_RECEIPT
        assert seen == [(tree.root.resolve(), "build")]
        for forbidden in ("HTTPS_PROXY", "LD_PRELOAD", "PYTHONPATH"):
            assert forbidden not in os.environ
        assert os.environ["OPTIMA_REBUILD_PHASE"] == "build"
        ordinary = json.loads(receipt.read_text())
        assert ordinary == {
            "build_spec_digest": required["OPTIMA_NATIVE_BUILD_SPEC_DIGEST"],
            "rebuild_applied": False,
            "schema": PREBUILD_SCHEMA,
            "stage_entries": [PREBUILD_RECEIPT],
            "target_architecture": "sm120",
            "tree_digest": tree.tree_digest,
        }
        assert receipt.read_bytes() == canonical_json_bytes(ordinary) + b"\n"
    finally:
        os.environ.clear()
        os.environ.update(original_environment)


def test_discovery_container_builds_fixed_overlay_and_conditional_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.metadata
    import optima.eval.oci_prebuild as prebuild_mod

    tree = _discovery_tree(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    stock_site = _stock_sglang(tmp_path)

    class Distribution:
        version = DEFAULT_DISCOVERY_POLICY.sglang_version

        @staticmethod
        def locate_file(path: str) -> Path:
            assert path == "sglang"
            return stock_site / "sglang"

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: Distribution())
    monkeypatch.setattr(prebuild_mod, "CONTAINER_TREE", str(tree.root))
    monkeypatch.setattr(prebuild_mod, "CONTAINER_STAGE", str(stage))
    original_environment = dict(os.environ)
    required = {
        "OPTIMA_NATIVE_BUILD_SPEC_DIGEST": _digest("discovery-build"),
        "OPTIMA_ENGINE_TREE_DIGEST": tree.tree_digest,
        "OPTIMA_TARGET_GPU_ARCH": "sm120",
        "OPTIMA_NATIVE_ARTIFACT_STAGE": str(stage),
        "OPTIMA_PINNED_BUILD_ROOTS": str(stock_site.resolve()),
        "OPTIMA_BUILD_PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "OPTIMA_BUILD_TMPDIR": "/tmp",
        "OPTIMA_NATIVE_COMPILE_TIMEOUT_S": "60",
        "OPTIMA_REBUILD_CONTAINER": "1",
    }
    try:
        os.environ.update(required)
        receipt_path = container_build()
    finally:
        os.environ.clear()
        os.environ.update(original_environment)

    receipt = json.loads(receipt_path.read_text())
    identity_digest = receipt["discovery_overlay_identity_digest"]
    assert receipt["stage_entries"] == ["dep_overlays", PREBUILD_RECEIPT]
    assert receipt_path.read_bytes() == canonical_json_bytes(receipt) + b"\n"
    publication = publish_native_artifact(
        stage,
        tmp_path / "publications",
        build_spec_digest=required["OPTIMA_NATIVE_BUILD_SPEC_DIGEST"],
    )
    overlay = reopen_discovery_overlay(
        publication,
        expected_identity_digest=identity_digest,
    )
    assert overlay.identity.proposal_digest == inspect_discovery(
        tree.root / "discovery"
    ).proposal_digest
    assert (overlay.site_root / "sglang/srt/layers/activation.py").read_text() == (
        "VALUE = 2\n"
    )


@pytest.mark.parametrize(
    "version,inside_root,match",
    (
        ("wrong-version", True, "version differs"),
        (DEFAULT_DISCOVERY_POLICY.sglang_version, False, "outside validator-pinned"),
    ),
)
def test_stock_sglang_distribution_must_match_version_and_pinned_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    inside_root: bool,
    match: str,
) -> None:
    import importlib.metadata
    import optima.eval.oci_prebuild as prebuild_mod

    stock_site = _stock_sglang(tmp_path)
    pinned = stock_site if inside_root else tmp_path / "other-root"
    pinned.mkdir(exist_ok=True)

    class Distribution:
        @staticmethod
        def locate_file(_path: str) -> Path:
            return stock_site / "sglang"

    distribution = Distribution()
    distribution.version = version
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    with pytest.raises(OCIPrebuildError, match=match):
        prebuild_mod._stock_sglang_site_root(
            expected_version=DEFAULT_DISCOVERY_POLICY.sglang_version,
            pinned_build_roots=(str(pinned.resolve()),),
        )


def test_stock_sglang_distribution_resolves_pinned_editable_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.metadata
    import importlib.util
    from types import SimpleNamespace

    import optima.eval.oci_prebuild as prebuild_mod

    source_site = _stock_sglang(tmp_path)
    metadata_site = tmp_path / "site-packages"
    metadata_site.mkdir()

    class Distribution:
        version = DEFAULT_DISCOVERY_POLICY.sglang_version

        @staticmethod
        def locate_file(_path: str) -> Path:
            return metadata_site / "sglang"

    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: Distribution())
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(
            origin=str(source_site / "sglang" / "__init__.py"),
            submodule_search_locations=(str(source_site / "sglang"),),
        ),
    )

    assert prebuild_mod._stock_sglang_site_root(
        expected_version=DEFAULT_DISCOVERY_POLICY.sglang_version,
        pinned_build_roots=(str(source_site.resolve()),),
    ) == source_site.resolve()


def test_seccomp_bytes_and_existing_publication_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tree_row, launch, binding, _preflight_row, config = _case(tmp_path)
    config.seccomp_profile.write_text("tampered\n")
    with pytest.raises(OCIPrebuildError, match="seccomp"):
        run_oci_prebuild(launch, binding, config)

    # A destination occupying the canonical address is validated, never repaired.
    config.seccomp_profile.write_text('{"defaultAction":"SCMP_ACT_ERRNO"}\n')
    digest = binding.native_build_spec.digest
    destination = config.publication_root / digest[:2] / digest
    destination.mkdir(parents=True)
    (destination / "garbage").write_text("x")
    with pytest.raises(Exception, match="native artifact|mode|manifest"):
        run_oci_prebuild(launch, binding, config)


def test_materialized_dep_cuda_tree_builds_publishes_and_reopens_load_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Join the PR3a materializer to both PR2a native consumers.

    The compiler and native loader are the only synthetic edges.  Everything around
    them is production code: contribution inspection and namespacing, rebuild-plan
    resolution/order, dependency patching, CUDA source discovery and receipts,
    immutable host publication, load-only full-publication reopen, and the
    FlashInfer load-only generator installed from the published artifact.
    """

    import functools
    import sys
    import types
    from dataclasses import asdict

    import optima.dep_policy as dep_policy
    import optima.eval.oci_prebuild as prebuild_module
    import optima.integrations.flashinfer_overlay as flashinfer_overlay
    import optima.patchers.build_cuda_ext as cuda_patcher
    import optima.rebuild as rebuild
    from optima.bundle_hash import content_hash
    from optima.engine_tree import inspect_contribution
    from optima.eval.native_artifact import publish_native_artifact
    from optima.stack_manifest import ProposalContributionRef

    fixture = Path(__file__).parent / "fixtures" / "stack_fused_epilogue_atomic"
    catalog = default_target_catalog()
    snapshot = catalog.snapshot()
    context = EvaluationStackContext(
        runtime_digest=_digest("joined-runtime"),
        base_engine_digest=_digest("joined-base"),
        arena_digest=_digest("joined-arena"),
        catalog_snapshot=snapshot,
        catalog_digest=catalog.digest,
        target_spec_digests={
            row["target_id"]: catalog.target_spec_digest(row["target_id"])
            for row in snapshot["targets"]
        },
    )
    inspected = inspect_contribution(fixture, catalog=catalog)
    proposal = ProposalContributionRef(
        target_id=inspected.target_id,
        target_spec_digest=inspected.target_spec_digest,
        artifact_digest=content_hash(fixture),
        selected_payload_digest=inspected.selected_payload_digest,
        attribution_digest=_digest("joined-attribution"),
    )
    stack = EvaluationStackManifest(
        runtime_digest=context.runtime_digest,
        base_engine_digest=context.base_engine_digest,
        arena_digest=context.arena_digest,
        catalog_snapshot=snapshot,
        catalog_digest=catalog.digest,
        entries={proposal.target_id: proposal},
    )
    tree = materialize_engine_tree(
        stack,
        context=context,
        catalog=catalog,
        resolver={("proposal", proposal.artifact_digest): fixture},
        destination=tmp_path / "materialized",
    )
    manifest = __import__("optima.manifest", fromlist=["load_manifest"]).load_manifest(
        tree.root
    )
    assert manifest.bundle_id == "optima-materialized-v1"
    assert [step["path"] for step in json.loads((tree.root / "rebuild.json").read_text())["steps"]] == [
        "optima/patchers/apply_dep_patch.py",
        "optima/patchers/build_cuda_ext.py",
    ]
    assert manifest.dep_patches[0].path.startswith("patches/optima_c_")
    assert manifest.ops[0].cuda_sources[0].startswith("cuda/optima_c_")

    # A minimal image-owned FlashInfer source tree matching the policy-valid patch.
    image_root = tmp_path / "image-root"
    dependency_source = (
        image_root / "flashinfer/data/csrc/fused_moe/fused_moe.cu"
    )
    dependency_source.parent.mkdir(parents=True)
    dependency_source.write_text("old_finalize();\n")
    monkeypatch.setattr(
        dep_policy, "dependency_site_root", lambda _policy: image_root
    )

    apply_script = Path(__file__).parents[1] / "optima/patchers/apply_dep_patch.py"
    apply_namespace: dict[str, object] = {}
    exec(
        compile(
            apply_script.read_text().replace("\nmain()\n", "\n"),
            str(apply_script),
            "exec",
        ),
        apply_namespace,
    )
    prebuilt_source = tmp_path / "fused_moe_103.so"
    prebuilt_source.write_bytes(b"synthetic-flashinfer-module")

    def fake_prebuilt_modules(policy, *, target, overlay_subtree, artifact_root):
        assert (overlay_subtree / "fused_moe/fused_moe.cu").read_text() == (
            "export_prefinalize();\n"
        )
        rows = []
        for module in policy.prebuilt_modules:
            relative = dep_policy.prebuilt_module_relative_path(target, module)
            destination = artifact_root / relative
            digest, size = apply_namespace["_copy_built_module"](
                prebuilt_source, destination
            )
            rows.append(
                {
                    **asdict(module),
                    "path": relative,
                    "sha256": digest,
                    "size": size,
                }
            )
        return rows

    apply_namespace["_build_prebuilt_modules"] = fake_prebuilt_modules

    compiler_environment = {
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "SOURCE_DATE_EPOCH": "0",
        "TEMP": "/tmp",
        "TMP": "/tmp",
        "TMPDIR": "/tmp",
        "TZ": "UTC",
    }
    build_context = {
        "compiler_env_digest": _digest("joined-compiler-env"),
        "cxx11_abi": 1,
        "link_libraries": list(cuda_patcher._LINK_LIBRARIES),
        "nvcc": {
            "path": "/usr/local/cuda/bin/nvcc",
            "sha256": _digest("joined-nvcc"),
            "version": "synthetic nvcc",
        },
        "nvcc_architecture": "sm_103a",
        "nvcc_flags": list(cuda_patcher._COMPILE_FLAGS),
        "pinned_build_roots": [str(image_root.resolve())],
        "ptxas": {
            "path": "/usr/local/cuda/bin/ptxas",
            "sha256": _digest("joined-ptxas"),
            "version": "synthetic ptxas",
        },
        "python_include": "/usr/include/python3.12",
        "python_soabi": "cpython-312-x86_64-linux-gnu",
        "python_version": "3.12.0",
        "torch_api_include": "/image/torch/api/include",
        "torch_cuda_version": "13.0",
        "torch_include": "/image/torch/include",
        "torch_lib": "/image/torch/lib",
        "torch_version": "2.synthetic",
    }
    monkeypatch.setattr(
        cuda_patcher, "_compiler_environment", lambda: dict(compiler_environment)
    )
    monkeypatch.setattr(
        cuda_patcher,
        "_build_context",
        lambda architecture, env, production: dict(build_context),
    )

    def fake_compile(*, bundle, source, output, depfile, module_name, context, env):
        assert Path(bundle) == tree.root.resolve()
        assert source.startswith("cuda/optima_c_")
        assert env == compiler_environment
        output.write_bytes((module_name + ":synthetic-cuda-extension").encode())
        depfile.write_text(f"{module_name}: {source}\n")

    monkeypatch.setattr(cuda_patcher, "_compile", fake_compile)
    native_loads: list[tuple[str, str, Path]] = []
    monkeypatch.setattr(
        cuda_patcher, "_load", lambda *args: native_loads.append(args)
    )

    patcher_phases: list[tuple[str, str]] = []

    def run_reviewed_patcher(path, *, run_name):
        name = Path(path).name
        patcher_phases.append((name, os.environ["OPTIMA_REBUILD_PHASE"]))
        if name == "apply_dep_patch.py":
            apply_namespace["main"]()
        elif name == "build_cuda_ext.py":
            cuda_patcher.main()
        else:  # pragma: no cover - rebuild parser constrains the registry
            raise AssertionError(f"unexpected patcher {name}")
        return {}

    monkeypatch.setattr(rebuild.runpy, "run_path", run_reviewed_patcher)

    stage = tmp_path / "native-stage"
    stage.mkdir()
    build_spec_digest = _digest("joined-native-build")
    monkeypatch.setenv("OPTIMA_REBUILD_CONTAINER", "1")
    monkeypatch.setenv("OPTIMA_NATIVE_ARTIFACT_STAGE", str(stage))
    monkeypatch.setenv("OPTIMA_NATIVE_BUILD_SPEC_DIGEST", build_spec_digest)
    monkeypatch.setenv("OPTIMA_ENGINE_TREE_DIGEST", tree.tree_digest)
    monkeypatch.setenv("OPTIMA_TARGET_GPU_ARCH", "sm103")
    monkeypatch.setenv("OPTIMA_PINNED_BUILD_ROOTS", str(image_root))
    monkeypatch.setenv("OPTIMA_BUILD_PATH", compiler_environment["PATH"])
    monkeypatch.setenv("OPTIMA_BUILD_TMPDIR", "/tmp")
    monkeypatch.setenv("OPTIMA_NATIVE_COMPILE_TIMEOUT_S", "60")

    monkeypatch.setattr(prebuild_module, "CONTAINER_TREE", str(tree.root.resolve()))
    monkeypatch.setattr(prebuild_module, "CONTAINER_STAGE", str(stage.resolve()))
    original_environment = dict(os.environ)
    try:
        receipt = prebuild_module.container_build()
    finally:
        os.environ.clear()
        os.environ.update(original_environment)
    assert receipt == stage / PREBUILD_RECEIPT
    assert {path.name for path in stage.iterdir()} == {
        "cuda",
        "dep_modules",
        "dep_overlays",
        PREBUILD_RECEIPT,
    }
    assert native_loads == []

    publication = publish_native_artifact(
        stage,
        tmp_path / "publications",
        build_spec_digest=build_spec_digest,
    )
    monkeypatch.setenv("OPTIMA_ENGINE_WORKER", "1")
    monkeypatch.setenv("OPTIMA_REBUILD_PHASE", "load")
    monkeypatch.setenv("OPTIMA_BUNDLE_PATH", str(tree.root))
    monkeypatch.setenv("OPTIMA_NATIVE_ARTIFACT_ROOT", str(publication.root))
    monkeypatch.setenv(
        "OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST",
        publication.publication_digest,
    )

    assert rebuild.apply_rebuild_plan(tree.root, phase="load")
    assert len(native_loads) == 1
    alias, module_name, artifact = native_loads[0]
    assert alias.startswith("optima_c_")
    assert module_name.startswith("optima_cuda_")
    assert publication.root in artifact.parents

    environment = types.ModuleType("flashinfer.jit.env")
    environment.FLASHINFER_CSRC_DIR = Path("/stock/csrc")
    generator_module = types.ModuleType("flashinfer.jit.fused_moe")
    generator_module.gen_cutlass_fused_moe_sm103_module = lambda use_fast_build=False: (
        "stock",
        use_fast_build,
    )
    consumer = types.ModuleType("flashinfer.fused_moe.core")
    consumer.gen_cutlass_fused_moe_sm103_module = (
        generator_module.gen_cutlass_fused_moe_sm103_module
    )

    @functools.cache
    def cached_module(_backend):
        return "stock"

    consumer.get_cutlass_fused_moe_module = cached_module
    tvm_loads: list[str] = []
    tvm_ffi = types.ModuleType("tvm_ffi")
    tvm_ffi.load_module = lambda path: tvm_loads.append(path) or f"loaded:{path}"
    for name, module in (
        ("flashinfer.jit.env", environment),
        ("flashinfer.jit.fused_moe", generator_module),
        ("flashinfer.fused_moe.core", consumer),
        ("tvm_ffi", tvm_ffi),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(flashinfer_overlay, "_installed", False)
    monkeypatch.setenv("OPTIMA_ACTIVE", "1")

    flashinfer_overlay.install(registry=None)
    expected_overlay = (
        publication.root
        / "dep_overlays/flashinfer/flashinfer/data/csrc"
    )
    expected_module = (
        publication.root
        / "dep_modules/flashinfer/fused_moe_103/fused_moe_103.so"
    )
    assert environment.FLASHINFER_CSRC_DIR == expected_overlay
    load_only = consumer.gen_cutlass_fused_moe_sm103_module(False)
    assert not hasattr(load_only, "build")
    assert load_only.build_and_load() == f"loaded:{expected_module}"
    assert tvm_loads == [str(expected_module)]
    assert patcher_phases == [
        ("apply_dep_patch.py", "build"),
        ("build_cuda_ext.py", "build"),
        ("apply_dep_patch.py", "load"),
        ("build_cuda_ext.py", "load"),
    ]
