from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import optima.release_host as host
from optima.release import (
    ContainerReproducibility,
    sign_container_reproducibility,
    verify_container_reproducibility,
)


def _d(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _raw(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class _Response:
    def __init__(self, url: str, payload: bytes, *, digest: str | None = None):
        self.status = 200
        self._url = url
        self._payload = payload
        self.headers = {}
        if digest is not None:
            self.headers["Docker-Content-Digest"] = digest

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def geturl(self):
        return self._url

    def getcode(self):
        return self.status

    def read(self, limit: int):
        return self._payload[:limit]


def _registry_objects(*, descriptor: str | None = None, platform: str = "linux/amd64"):
    os_name, architecture = platform.split("/")
    layer_diff = _d("uncompressed-layer")
    config = _raw(
        {
            "architecture": architecture,
            "config": {
                "Cmd": ["serve"],
                "Entrypoint": ["python"],
                "Env": ["A=B"],
                "Labels": {
                    "org.optima.release.descriptor": descriptor or _d("release"),
                    "org.optima.seccomp.sha256": _d("seccomp"),
                    "org.optima.runtime-overlays": _d("overlay"),
                },
            },
            "os": os_name,
            "rootfs": {"diff_ids": [f"sha256:{layer_diff}"], "type": "layers"},
        }
    )
    config_digest = hashlib.sha256(config).hexdigest()
    manifest = _raw(
        {
            "config": {
                "digest": f"sha256:{config_digest}",
                "mediaType": "application/vnd.oci.image.config.v1+json",
                "size": len(config),
            },
            "layers": [
                {
                    "digest": f"sha256:{_d('compressed-layer')}",
                    "mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                    "size": 123,
                }
            ],
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "schemaVersion": 2,
        }
    )
    return manifest, config


def _opener(manifest: bytes, config: bytes, *, header_digest: str | None = None):
    manifest_digest = hashlib.sha256(manifest).hexdigest()

    def open_(request, timeout):
        assert timeout == 30
        if "/manifests/" in request.full_url:
            digest = header_digest or f"sha256:{manifest_digest}"
            return _Response(request.full_url, manifest, digest=digest)
        assert "/blobs/sha256:" in request.full_url
        return _Response(request.full_url, config)

    return open_


def _image(*, descriptor: str | None = None, image_digest: str | None = None,
           platform: str = "linux/amd64") -> host.RegistryImage:
    return host.RegistryImage(
        "registry.example/optima/engine",
        "build1",
        image_digest or _d("image"),
        "application/vnd.oci.image.manifest.v1+json",
        host.OCIDescriptor(
            "application/vnd.oci.image.config.v1+json", _d("config"), 100
        ),
        (
            host.OCIDescriptor(
                "application/vnd.oci.image.layer.v1.tar+gzip", _d("layer"), 1000
            ),
        ),
        platform,
        {
            "org.optima.release.descriptor": descriptor or _d("release"),
            "org.optima.seccomp.sha256": _d("seccomp"),
            "org.optima.runtime-overlays": _d("overlay"),
        },
        ("python",),
        ("serve",),
        ("A=B",),
    )


def test_registry_v2_reopens_raw_manifest_and_config() -> None:
    manifest, config = _registry_objects()
    client = host.RegistryV2Client(
        "registry.example/optima/engine", opener=_opener(manifest, config)
    )
    image = client.reopen("build1", expected_platform="linux/amd64")
    assert image.manifest_digest == hashlib.sha256(manifest).hexdigest()
    assert image.config.sha256 == hashlib.sha256(config).hexdigest()
    assert image.platform == "linux/amd64"
    assert image.entrypoint == ("python",)
    assert image.command == ("serve",)


def test_registry_v2_rejects_wrong_digest_and_platform() -> None:
    manifest, config = _registry_objects()
    client = host.RegistryV2Client(
        "registry.example/optima/engine",
        opener=_opener(manifest, config, header_digest="sha256:" + _d("wrong")),
    )
    with pytest.raises(host.ReleaseHostError, match="digest header"):
        client.reopen("build1", expected_platform="linux/amd64")
    client = host.RegistryV2Client(
        "registry.example/optima/engine", opener=_opener(manifest, config)
    )
    with pytest.raises(host.ReleaseHostError, match="platform differs"):
        client.reopen("build1", expected_platform="linux/arm64")


def test_registry_v2_rejects_unverified_descriptor_shapes() -> None:
    manifest, config = _registry_objects()
    value = json.loads(manifest)
    value["layers"][0]["urls"] = ["https://untrusted.example/layer"]
    malformed = _raw(value)
    client = host.RegistryV2Client(
        "registry.example/optima/engine", opener=_opener(malformed, config)
    )
    with pytest.raises(host.ReleaseHostError, match="descriptor fields"):
        client.reopen("build1", expected_platform="linux/amd64")


def test_double_build_uses_closed_argv_and_signs_actual_common_digest(
    monkeypatch, tmp_path: Path
) -> None:
    descriptor_digest = _d("release")
    descriptor = SimpleNamespace(
        digest=descriptor_digest,
        serve=SimpleNamespace(oci_platform="linux/amd64"),
        seccomp=SimpleNamespace(sha256=_d("seccomp")),
    )
    release = SimpleNamespace(descriptor=descriptor)
    monkeypatch.setattr(host, "_reopen_context", lambda *_args, **_kwargs: (tmp_path, release))
    monkeypatch.setattr(host, "_overlay_digest", lambda _release: _d("overlay"))
    reopened = []

    class Registry:
        def __init__(self, *_args, **_kwargs):
            pass

        def reopen(self, tag, *, expected_platform):
            reopened.append(tag)
            return replace(_image(), reference=tag)

    monkeypatch.setattr(host, "RegistryV2Client", Registry)
    commands = []
    signing_key = b"\x33" * 32
    trusted_key = sign_container_reproducibility(
        ContainerReproducibility(
            descriptor_digest, _d("image"), _d("image"), "linux/amd64"
        ),
        signing_key,
    ).signature.public_key

    def runner(argv, **kwargs):
        commands.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    signed = host.publish_container_twice(
        tmp_path,
        repository="registry.example/optima/engine",
        expected_descriptor_digest=descriptor_digest,
        expected_public_key=trusted_key,
        signing_private_key=signing_key,
        runner=runner,
    )
    verify_container_reproducibility(
        signed, expected_public_key=signed.signature.public_key
    )
    assert signed.attestation.first_oci_digest == _d("image")
    assert len(commands) == 2 and len(set(reopened)) == 2
    for argv, kwargs in commands:
        assert argv[:5] == (
            "/usr/local/bin/docker", "buildx", "build", "--builder", "optima-release-builder"
        )
        assert "--no-cache" in argv and "--output" in argv
        assert argv[argv.index("--network") + 1] == "none"
        assert kwargs["shell"] is False


def _fake_release(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    seccomp = b'{"defaultAction":"SCMP_ACT_ERRNO"}'
    path = artifact_root / "seccomp.json"
    path.write_bytes(seccomp)
    descriptor = SimpleNamespace(
        digest=_d("release"),
        serve=SimpleNamespace(oci_platform="linux/amd64", model_mount="/models/M3"),
        seccomp=SimpleNamespace(
            name="seccomp.json", sha256=hashlib.sha256(seccomp).hexdigest()
        ),
    )
    return SimpleNamespace(root=tmp_path, descriptor=descriptor), path


def _authorized(tmp_path: Path, monkeypatch):
    release, seccomp_path = _fake_release(tmp_path)
    monkeypatch.setattr(host, "_overlay_digest", lambda _release: _d("overlay"))
    image = replace(
        _image(descriptor=release.descriptor.digest),
        labels={
            "org.optima.release.descriptor": release.descriptor.digest,
            "org.optima.seccomp.sha256": release.descriptor.seccomp.sha256,
            "org.optima.runtime-overlays": _d("overlay"),
        },
    )
    key = b"\x44" * 32
    signed = sign_container_reproducibility(
        ContainerReproducibility(
            release.descriptor.digest,
            image.manifest_digest,
            image.manifest_digest,
            image.platform,
        ),
        key,
    )
    monkeypatch.setattr(host, "reopen_release", lambda *_args, **_kwargs: release)
    authorized = host.authorize_release_container(
        tmp_path,
        signed,
        image,
        repository=image.repository,
        expected_public_key=signed.signature.public_key,
        expected_descriptor_digest=release.descriptor.digest,
        expected_platform=image.platform,
        expected_image_digest=image.manifest_digest,
    )
    assert authorized.seccomp_path == seccomp_path.resolve()
    return authorized, signed, image, release


def test_authorization_rejects_wrong_key_digest_and_label(
    tmp_path: Path, monkeypatch
) -> None:
    authorized, signed, image, release = _authorized(tmp_path, monkeypatch)
    assert authorized.image_reference.endswith(image.manifest_digest)
    with pytest.raises(Exception, match="trusted release authority"):
        host.authorize_release_container(
            tmp_path,
            signed,
            image,
            repository=image.repository,
            expected_public_key="00" * 32,
            expected_descriptor_digest=release.descriptor.digest,
            expected_platform=image.platform,
            expected_image_digest=image.manifest_digest,
        )
    with pytest.raises(host.ReleaseHostError, match="host expectations"):
        host.authorize_release_container(
            tmp_path,
            signed,
            image,
            repository=image.repository,
            expected_public_key=signed.signature.public_key,
            expected_descriptor_digest=release.descriptor.digest,
            expected_platform=image.platform,
            expected_image_digest=_d("other-image"),
        )
    wrong_label = replace(
        image,
        labels={
            "org.optima.release.descriptor": _d("wrong-release"),
            "org.optima.seccomp.sha256": release.descriptor.seccomp.sha256,
            "org.optima.runtime-overlays": _d("overlay"),
        },
    )
    with pytest.raises(host.ReleaseHostError, match="registry image differs"):
        host.authorize_release_container(
            tmp_path,
            signed,
            wrong_label,
            repository=image.repository,
            expected_public_key=signed.signature.public_key,
            expected_descriptor_digest=release.descriptor.digest,
            expected_platform=image.platform,
            expected_image_digest=image.manifest_digest,
        )


def _image_inspect(authorized: host.AuthorizedReleaseContainer) -> dict:
    image = authorized.image
    return {
        "Architecture": "amd64",
        "Config": {
            "Cmd": list(image.command),
            "Entrypoint": list(image.entrypoint),
            "Env": list(image.environment),
            "Labels": dict(image.labels),
            "Volumes": None,
        },
        "Id": f"sha256:{image.config.sha256}",
        "Os": "linux",
        "RepoDigests": [authorized.image_reference],
    }


def _container_inspect(
    authorized: host.AuthorizedReleaseContainer, model: Path, cid: str, name: str
) -> dict:
    image = authorized.image
    destination = authorized.release.descriptor.serve.model_mount
    return {
        "Config": {
            "Cmd": list(image.command),
            "Entrypoint": list(image.entrypoint),
            "Env": list(image.environment),
            "Image": authorized.image_reference,
            "Labels": dict(image.labels),
        },
        "HostConfig": {
            "Binds": None,
            "CapAdd": None,
            "CapDrop": ["ALL"],
            "DeviceRequests": [
                {
                    "Capabilities": [["gpu"]],
                    "Count": -1,
                    "DeviceIDs": None,
                    "Driver": "",
                    "Options": {},
                }
            ],
            "IpcMode": "host",
            "Mounts": [
                {
                    "ReadOnly": True,
                    "Source": str(model),
                    "Target": destination,
                    "Type": "bind",
                }
            ],
            "NetworkMode": "host",
            "Privileged": False,
            "ReadonlyRootfs": True,
            "SecurityOpt": [
                "no-new-privileges=true",
                f"seccomp={authorized.seccomp_path}",
            ],
            "ShmSize": 32 << 30,
            "Tmpfs": {
                "/tmp": "rw,nosuid,nodev,noexec,size=17179869184,mode=1777"
            },
        },
        "Id": cid,
        "Image": f"sha256:{image.config.sha256}",
        "Mounts": [
            {
                "Destination": destination,
                "RW": False,
                "Source": str(model),
                "Type": "bind",
            }
        ],
        "Name": f"/{name}",
        "State": {"Running": False, "Status": "created"},
    }


def test_create_inspects_before_start_and_uses_closed_lifecycle(
    tmp_path: Path, monkeypatch
) -> None:
    release_root = tmp_path / "release"
    release_root.mkdir()
    authorized, _, _, _ = _authorized(release_root, monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    cid = "a" * 64
    name = "optima-release"
    calls = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        if argv[1:3] == ("image", "inspect"):
            payload = [_image_inspect(authorized)]
        elif argv[1:3] == ("container", "create"):
            return subprocess.CompletedProcess(argv, 0, (cid + "\n").encode(), b"")
        elif argv[1:3] == ("container", "inspect"):
            payload = [_container_inspect(authorized, model, cid, name)]
        else:
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        return subprocess.CompletedProcess(argv, 0, _raw(payload), b"")

    handle = host.create_release_container(
        authorized, model, name=name, runner=runner
    )
    create = next(argv for argv in calls if argv[1:3] == ("container", "create"))
    assert "--read-only" in create and ("--gpus", "all") == create[create.index("--gpus"):create.index("--gpus") + 2]
    assert create[-1] == authorized.image_reference
    handle.start()
    receipts = handle.copy_receipts(tmp_path / "receipts")
    assert receipts.is_dir()
    handle.destroy()
    assert any(argv[1:3] == ("container", "start") for argv in calls)
    assert any(argv[1:3] == ("container", "cp") for argv in calls)
    assert any(argv[1:3] == ("container", "rm") for argv in calls)


def test_create_destroys_container_when_inspect_policy_differs(
    tmp_path: Path, monkeypatch
) -> None:
    release_root = tmp_path / "release"
    release_root.mkdir()
    authorized, _, _, _ = _authorized(release_root, monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    cid = "b" * 64
    calls = []

    def runner(argv, **_kwargs):
        calls.append(argv)
        if argv[1:3] == ("image", "inspect"):
            payload = [_image_inspect(authorized)]
        elif argv[1:3] == ("container", "create"):
            return subprocess.CompletedProcess(argv, 0, (cid + "\n").encode(), b"")
        elif argv[1:3] == ("container", "inspect"):
            row = _container_inspect(authorized, model, cid, "optima-release")
            row["HostConfig"]["ReadonlyRootfs"] = False
            payload = [row]
        else:
            return subprocess.CompletedProcess(argv, 0, b"", b"")
        return subprocess.CompletedProcess(argv, 0, _raw(payload), b"")

    with pytest.raises(host.ReleaseHostError, match="closed host policy"):
        host.create_release_container(
            authorized, model, name="optima-release", runner=runner
        )
    assert calls[-1][1:5] == ("container", "rm", "--force", "--volumes")
