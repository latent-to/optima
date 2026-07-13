from __future__ import annotations

import hashlib
import json
import os
import stat
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from optima.eval.evidence_store import EvidenceArtifactRef
from optima.engine_tree import (
    EngineTreeError,
    inspect_contribution,
    integrated_source_tree_digest,
    materialize_engine_tree,
)
from optima.eval.engine_launch import (
    NativeBuildSpec,
    native_compiler_policy_digest,
    native_patcher_digest,
    native_toolchain_digest,
)
from optima.eval.calibration import (
    CalibrationContext,
    CalibrationControl,
    CalibrationManifest,
    MetricCalibration,
    SpeedCalibration,
)
from optima.eval.native_artifact import publish_native_artifact
from optima.eval.qualification import ReferenceManifest
from optima.model_provision import provision_model
from optima.release import (
    ContainerReproducibility,
    EngineReleaseDescriptor,
    ReleaseArtifact,
    ReleaseError,
    ReleaseSignature,
    ServeSpec,
    SignedContainerReproducibility,
    container_context,
    prepare_release,
    publish_release,
    reopen_release,
    sign_container_reproducibility,
    sign_release,
    verify_container_reproducibility,
    verify_release_signature,
    verify_serve_receipts,
)
from optima.release_runtime import (
    ReleaseRuntimeError,
    _closed_serving_environment,
    verify_serving_release,
)
from optima.release_host import _reopen_context
from optima.stack_identity import canonical_json_bytes
from optima.stack_manifest import (
    EngineReleaseManifest,
    IntegrationReviewArtifacts,
    IntegrationReviewRecord,
    ReleaseStackContext,
)
from optima.target_catalog import default_target_catalog


ROOT = Path(__file__).resolve().parents[1]
MSA = ROOT / "tests/fixtures/stack_msa_singleton"


def _d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _reference(manifest, model) -> ReferenceManifest:
    return ReferenceManifest(
        _d("pristine-stack"), _d("pristine-tree"), _d("pristine-launch"),
        manifest.runtime_digest, manifest.base_engine_digest, _d("arena"),
        manifest.catalog_digest, _d("controller"), _d("worker"),
        _d("model-revision"), _d("model-manifest"), model.receipt.content_digest,
        _d("logical-hardware"), _d("workload"), _d("tokenizer"),
        _d("hidden-corpus"), _d("hidden-judge"), _d("selection-policy"),
    )


def _calibration(reference: ReferenceManifest) -> CalibrationManifest:
    return CalibrationManifest(
        context=CalibrationContext(
            reference.digest,
            reference.arena_digest,
            reference.runtime_digest,
            reference.base_engine_digest,
            reference.model_revision_digest,
            reference.model_manifest_digest,
            reference.model_content_digest,
            reference.logical_hardware_digest,
            reference.workload_digest,
            _d("verification-policy"),
            reference.controller_distribution_digest,
        ),
        algorithm_id="teacher-familywise-v1",
        status="frozen",
        speed=SpeedCalibration("0.005", "2", "0.1"),
        quality_metrics=(
            MetricCalibration("mean_nll", "lower", "0.02", "0.01"),
            MetricCalibration("task_score", "higher", "0.03", "0.02", "0.8"),
        ),
        familywise_z="2.576",
        raw_evidence_digest=_d("calibration-raw"),
        seed_digests=tuple(sorted((_d("seed-a"), _d("seed-b")))),
        controls=(
            CalibrationControl("negative", _d("negative-seed"), _d("negative-raw"), "FAIL"),
            CalibrationControl("positive", _d("positive-seed"), _d("positive-raw"), "PASS"),
            CalibrationControl("stock", _d("stock-seed"), _d("stock-raw"), "PASS"),
        ),
    )


def _evidence(domain: str, digest: str) -> EvidenceArtifactRef:
    return EvidenceArtifactRef(
        domain, digest, 1, "application/json", f"optima.{domain}.v1"
    )


def _release_tree(tmp_path: Path):
    catalog = default_target_catalog()
    inspected = inspect_contribution(MSA, catalog=catalog)
    record = IntegrationReviewRecord(
        target_id=inspected.target_id,
        target_spec_digest=inspected.target_spec_digest,
        proposal_contribution_digest=_d("proposal"),
        settlement_candidate_digest=_d("candidate"),
        settlement_evidence_digest=_d("settlement-evidence"),
        crown_event_digest=_d("crown"),
        primary_attempt_digest=_d("primary"),
        reproduction_attempt_digest=_d("reproduction"),
        integrated_source_tree_digest=integrated_source_tree_digest(MSA),
        selected_payload_digest=inspected.selected_payload_digest,
        attribution_digest=_d("attribution"),
        license_evidence_digest=_d("license"),
        provenance_evidence_digest=_d("provenance"),
        security_review_digest=_d("security"),
        compatibility_evidence_digest=_d("compatibility"),
        test_evidence_digest=_d("tests"),
        artifacts=IntegrationReviewArtifacts(
            _evidence("qualification.cohort-attempt", _d("primary")),
            _evidence("qualification.cohort-attempt", _d("reproduction")),
            _evidence("integration.license", _d("license")),
            _evidence("integration.provenance", _d("provenance")),
            _evidence("integration.security-review", _d("security")),
            _evidence("integration.compatibility", _d("compatibility")),
            _evidence("integration.tests", _d("tests")),
        ),
        reviewer="optima-release-review",
        review_commit="a" * 40,
    )
    ref = record.integrated_ref()
    specs = {
        row["target_id"]: catalog.target_spec_digest(row["target_id"])
        for row in catalog.snapshot()["targets"]
    }
    context = ReleaseStackContext(
        runtime_digest=_d("runtime"),
        base_engine_digest=_d("base"),
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        target_spec_digests=specs,
    )
    manifest = EngineReleaseManifest(
        runtime_digest=context.runtime_digest,
        base_engine_digest=context.base_engine_digest,
        catalog_snapshot=catalog.snapshot(),
        catalog_digest=catalog.digest,
        entries={ref.target_id: ref},
    )
    tree = materialize_engine_tree(
        manifest,
        context=context,
        catalog=catalog,
        resolver={("integrated", ref.integrated_source_tree_digest): MSA},
        destination=tmp_path / "engine-tree",
        integration_records={record.target_id: record},
    )
    return manifest, record, tree


def _native(tmp_path: Path, tree_digest: str):
    image = _d("native-image")
    platform = _d("native-platform")
    worker = _d("native-worker")
    dependency = _d("native-dependency")
    spec = NativeBuildSpec(
        tree_digest=tree_digest,
        image_digest=image,
        platform_digest=platform,
        worker_distribution_digest=worker,
        toolchain_digest=native_toolchain_digest(
            image_digest=image, platform_digest=platform
        ),
        patcher_digest=native_patcher_digest(worker_distribution_digest=worker),
        compiler_flags_digest=native_compiler_policy_digest(
            image_digest=image,
            worker_distribution_digest=worker,
            dependency_policy_digest=dependency,
            target_architecture="sm120",
        ),
        target_architecture="sm120",
        dependency_policy_digest=dependency,
    )
    stage = tmp_path / "native-stage"
    stage.mkdir()
    (stage / "kernel.bin").write_bytes(b"sealed-native-product")
    publication_root = tmp_path / "native-publications"
    publication_root.mkdir()
    publication = publish_native_artifact(
        stage, publication_root, build_spec_digest=spec.digest
    )
    return spec, publication


def _prepared(tmp_path: Path):
    manifest, record, tree = _release_tree(tmp_path)
    native_spec, native_publication = _native(tmp_path, tree.tree_digest)
    model_root = (tmp_path / "models" / "MiniMax-M3-NVFP4").resolve()
    model_root.mkdir(parents=True)
    (model_root / "config.json").write_text('{"model_type":"minimax"}\n')
    (model_root / "weights.safetensors").write_bytes(b"model-weights")
    receipts = tmp_path / "model-receipts"
    receipts.mkdir()
    model = provision_model(model_root, receipts, workers=1)
    reference = _reference(manifest, model)
    serve = ServeSpec(
        "example/sglang@sha256:" + "d" * 64,
        "linux/amd64",
        model_root.as_posix(),
        8,
        ("--host", "0.0.0.0"),
        (("CUDA_DEVICE_MAX_CONNECTIONS", "1"),),
    )
    prepared = prepare_release(
        source_root=ROOT,
        release_manifest=manifest,
        engine_tree=tree,
        integrations=(record,),
        model_id="MiniMax-M3-NVFP4",
        model_revision="b" * 40,
        model_manifest_digest=reference.model_manifest_digest,
        model_provision=model,
        native_build_spec=native_spec,
        native_publication=native_publication,
        seccomp_payload=(ROOT / "optima/eval/seccomp_moby_v0_2_1.json").read_bytes(),
        reference_manifest_payload=canonical_json_bytes(reference.to_dict()) + b"\n",
        calibration_manifest_payload=canonical_json_bytes(_calibration(reference).to_dict()) + b"\n",
        upstream_repository="https://github.com/sgl-project/sglang",
        upstream_revision="56e290315b8fdb4c8c10f8e31360d9bc3d878633",
        sglang_version="0.0.0.dev1+g56e290315",
        serve=serve,
    )
    return prepared, tree, model_root


def _publish(tmp_path: Path, key: bytes = b"\x11" * 32):
    prepared, tree, model_root = _prepared(tmp_path)
    signature = sign_release(prepared.descriptor, key)
    published = publish_release(
        tmp_path / "published",
        prepared.descriptor,
        signature,
        tree,
        prepared.payloads,
        prepared.native_publication,
        expected_public_key=signature.public_key,
    )
    return prepared, published, signature, model_root


def test_runtime_source_and_wheel_double_build_and_exclude_subnet_code(tmp_path: Path) -> None:
    prepared, _tree, _model_root = _prepared(tmp_path)
    wheel = tmp_path / prepared.descriptor.runtime_wheel.name
    wheel.write_bytes(prepared.payloads[wheel.name])
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert {"optima/bootstrap.py", "optima/release_runtime.py", "optima_engine_bootstrap.pth"} <= names
        assert "optima/eval/native_artifact.py" in names
        assert not any(name.startswith("optima/chain/") for name in names)
        assert "optima/commit_reveal.py" not in names
        assert "optima/settlement.py" not in names
        assert "optima/cli.py" not in names


def test_integration_record_is_required_and_cannot_be_an_opaque_digest(tmp_path: Path) -> None:
    manifest, record, tree = _release_tree(tmp_path)
    manifest.validate_integrations({record.target_id: record})
    with pytest.raises(ValueError, match="differ"):
        manifest.validate_integrations(
            {record.target_id: replace(record, security_review_digest=_d("different"))}
        )
    with pytest.raises(EngineTreeError, match="integration records"):
        materialize_engine_tree(
            manifest,
            context=ReleaseStackContext(
                runtime_digest=manifest.runtime_digest,
                base_engine_digest=manifest.base_engine_digest,
                catalog_snapshot=manifest.catalog_snapshot,
                catalog_digest=manifest.catalog_digest,
                target_spec_digests={
                    row["target_id"]: default_target_catalog().target_spec_digest(row["target_id"])
                    for row in default_target_catalog().snapshot()["targets"]
                },
            ),
            catalog=default_target_catalog(),
            resolver={},
            destination=tmp_path / "unreviewed",
        )
    assert tree.stack_digest == manifest.digest


def test_signed_release_reopens_native_model_and_chain_free_context(tmp_path: Path, monkeypatch) -> None:
    prepared, published, signature, model_root = _publish(tmp_path)
    descriptor = prepared.descriptor
    reopened = reopen_release(
        published.root,
        expected_descriptor_digest=descriptor.digest,
        expected_public_key=signature.public_key,
    )
    assert reopened.descriptor == descriptor
    assert reopened.native_publication.publication_digest == descriptor.native.publication_digest
    assert stat.S_IMODE(reopened.root.stat().st_mode) == 0o555
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    verified = verify_serving_release(
        release_root=reopened.root,
        expected_public_key=signature.public_key,
        model_root=model_root,
        command=("/usr/bin/python3", "-m", "sglang.launch_server", *descriptor.serve.command_arguments),
        require_seccomp=False,
    )
    env = dict(verified.environment)
    assert env["OPTIMA_NATIVE_BUILD_SPEC_DIGEST"] == descriptor.native.build_spec_digest
    assert env["OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST"] == descriptor.native.publication_digest
    assert env["OPTIMA_NATIVE_ARTIFACT_ROOT"].endswith(descriptor.native.build_spec_digest)
    assert env["OPTIMA_REBUILD_PHASE"] == "load"
    assert env["OPTIMA_PREBUILT_ARTIFACTS"] == "1"
    assert env["OPTIMA_TARGET_GPU_ARCH"] == "sm120"
    assert env["OPTIMA_MSA_PREFILL_SEAM"] == "1"
    assert env["OPTIMA_ARFUSION_SEAM"] == "0"
    assert env["OPTIMA_ATTENTION_SEAM"] == "0"
    assert env["OPTIMA_COLLECTIVE_SEAM"] == "0"
    assert env["OPTIMA_MOE_SEAM"] == "0"
    assert "OPTIMA_NATIVE_ARTIFACT_STAGE" not in env
    assert "OPTIMA_NATIVE_COMPILE_TIMEOUT_S" not in env
    closed = _closed_serving_environment(
        verified,
        {
            "PATH": "/host-controlled",
            "AWS_SECRET_ACCESS_KEY": "must-not-cross",
            "NVIDIA_VISIBLE_DEVICES": "all",
        },
    )
    assert closed["PATH"] == "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
    assert closed["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert "AWS_SECRET_ACCESS_KEY" not in closed

    context = container_context(
        reopened, tmp_path / "container-context",
        expected_public_key=signature.public_key,
    )
    dockerfile = (context / "Dockerfile").read_text()
    assert descriptor.serve.base_image in dockerfile
    assert "bittensor" not in dockerfile.lower()
    assert "wallet" not in dockerfile.lower()
    assert "OPTIMA_ACTIVE=1" not in dockerfile
    assert "optima.release_runtime" in dockerfile
    assert "install-reviewed-overlays" in dockerfile
    assert "org.optima.runtime-overlays" in dockerfile
    assert 'ENTRYPOINT ["/usr/bin/python3"' in dockerfile
    assert '"--model-path"' in dockerfile and '"--tp-size"' in dockerfile
    deployment = json.loads((context / "deployment.json").read_bytes())
    assert deployment["required_seccomp_profile"] == "seccomp.json"
    assert deployment["required_read_only_rootfs"] is True
    assert _reopen_context(
        context,
        expected_descriptor_digest=descriptor.digest,
        expected_public_key=signature.public_key,
    )[1].descriptor == descriptor


def test_public_key_model_native_and_canonical_evidence_fail_closed(tmp_path: Path) -> None:
    prepared, tree, model_root = _prepared(tmp_path)
    signature = sign_release(prepared.descriptor, b"\x22" * 32)
    with pytest.raises(ReleaseError, match="trusted release authority"):
        verify_release_signature(
            prepared.descriptor,
            signature,
            expected_public_key="00" * 32,
        )
    with pytest.raises(ReleaseError, match="native build product"):
        replace(prepared.descriptor, engine_tree_digest=_d("another-engine-tree"))
    bad_payloads = prepared.payloads
    bad_payloads[prepared.descriptor.sbom.name] = b"{}\n"
    with pytest.raises(ReleaseError, match="(?:SBOM|sbom)"):
        publish_release(
            tmp_path / "bad-published",
            prepared.descriptor,
            signature,
            tree,
            bad_payloads,
            prepared.native_publication,
            expected_public_key=signature.public_key,
        )
    published = publish_release(
        tmp_path / "published",
        prepared.descriptor,
        signature,
        tree,
        prepared.payloads,
        prepared.native_publication,
        expected_public_key=signature.public_key,
    )
    with pytest.raises(ReleaseError, match="trusted release authority"):
        reopen_release(published.root, expected_public_key="00" * 32)
    os.chmod(model_root / "weights.safetensors", 0o644)
    (model_root / "weights.safetensors").write_bytes(b"changed")
    with pytest.raises(ReleaseRuntimeError, match="model verification failed"):
        verify_serving_release(
            release_root=published.root,
            expected_public_key=signature.public_key,
            model_root=model_root,
            command=(
                "/usr/bin/python3", "-m", "sglang.launch_server",
                *prepared.descriptor.serve.command_arguments,
            ),
            require_seccomp=False,
        )


def test_serve_spec_reserves_model_tp_runtime_and_injection_environment() -> None:
    base = ServeSpec(
        "example/sglang@sha256:" + "d" * 64,
        "linux/amd64", "/models/m", 8, ("--host", "0.0.0.0"), (),
    )
    assert base.command_arguments[:4] == ("--model-path", "/models/m", "--tp-size", "8")
    for arguments in (("--model-path", "/other"), ("--tp-size=1",)):
        with pytest.raises(ReleaseError, match="override"):
            replace(base, engine_arguments=arguments)
    for key in (
        "OPTIMA_ACTIVE", "PYTHONPATH", "LD_PRELOAD", "SGLANG_PLUGINS", "PATH",
    ):
        with pytest.raises(ReleaseError, match="override"):
            replace(base, environment=((key, "x"),))


def test_release_artifact_roles_and_runtime_command_are_exact(tmp_path: Path, monkeypatch) -> None:
    prepared, published, signature, model_root = _publish(tmp_path)
    with pytest.raises(ReleaseError, match="wrong name/media"):
        replace(
            prepared.descriptor,
            seccomp=ReleaseArtifact.from_bytes(
                "renamed.json", "application/vnd.optima.seccomp+json", b"{}"
            ),
        )
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    with pytest.raises(ReleaseRuntimeError, match="command differs"):
        verify_serving_release(
            release_root=published.root,
            expected_public_key=signature.public_key,
            model_root=model_root,
            command=("/usr/bin/python3", "-m", "sglang.launch_server", "--tp-size", "1"),
            require_seccomp=False,
        )


def test_container_double_build_attestation_is_signed_by_trusted_authority() -> None:
    key = b"\x33" * 32
    receipt = ContainerReproducibility(
        _d("descriptor"), _d("oci"), _d("oci"), "linux/amd64"
    )
    signed = sign_container_reproducibility(receipt, key)
    assert isinstance(SignedContainerReproducibility.from_dict(signed.to_dict()), SignedContainerReproducibility)
    verify_container_reproducibility(signed, expected_public_key=signed.signature.public_key)
    with pytest.raises(ReleaseError, match="trusted release authority"):
        verify_container_reproducibility(signed, expected_public_key="00" * 32)
    with pytest.raises(ReleaseError, match="did not reproduce"):
        ContainerReproducibility(
            _d("descriptor"), _d("first"), _d("second"), "linux/amd64"
        )


def test_release_commands_require_independent_trusted_public_key() -> None:
    from optima.cli import build_parser

    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    for command in ("release-verify", "release-context"):
        action = next(
            row
            for row in subparsers.choices[command]._actions
            if "--expected-public-key" in row.option_strings
        )
        assert action.required is True
        options = {
            option
            for row in subparsers.choices[command]._actions
            for option in row.option_strings
        }
        assert not any(
            token in option
            for option in options
            for token in ("wallet", "hotkey", "netuid", "network", "miner")
        )


def test_signed_release_mode_rejects_arbitrary_namespace_root(monkeypatch, tmp_path: Path) -> None:
    import optima.seam as seam

    monkeypatch.setenv("OPTIMA_RELEASE_REQUIRED", "1")
    monkeypatch.setenv("OPTIMA_RELEASE_DESCRIPTOR_DIGEST", _d("release"))
    monkeypatch.setenv("OPTIMA_RELEASE_VERIFIED", _d("release"))
    monkeypatch.setenv("OPTIMA_ENGINE_WORKER", "1")
    monkeypatch.setenv("OPTIMA_BUNDLE_PATH", str(tmp_path / "engine-tree"))
    monkeypatch.setenv("OPTIMA_ENGINE_TREE_DIGEST", _d("tree"))
    monkeypatch.setenv("OPTIMA_STACK_DIGEST", _d("stack"))
    with pytest.raises(SystemExit, match="materialized namespace"):
        seam.activate()


def test_serve_receipts_require_active_routed_completed_per_rank_and_no_fallback(tmp_path: Path) -> None:
    _prepared_release, published, _signature, _model_root = _publish(tmp_path)
    receipt_root = tmp_path / "serve-receipts"
    receipt_root.mkdir()
    slot = "attention.msa_prefill_block_score"
    for rank in range(8):
        identity = {"pid": 1000 + rank, "rank": rank, "world_size": 8}
        (receipt_root / f"active.{rank}.json").write_text(
            json.dumps({**identity, "bundle": "/optima/engine-tree", "slots": [slot]})
        )
        for kind in ("fired", "completed"):
            (receipt_root / f"{kind}.{slot}.{rank}.json").write_text(
                json.dumps({**identity, "slot": slot})
            )
    verified = verify_serve_receipts(published, receipt_root)
    assert verified.expected_ranks == 8
    assert verified.expected_slots == (slot,)

    (receipt_root / f"fallback.{slot}.0.json").write_text(
        json.dumps({"pid": 1000, "rank": 0, "world_size": 8, "slot": slot})
    )
    with pytest.raises(ReleaseError, match="load failure or fallback"):
        verify_serve_receipts(published, receipt_root)
