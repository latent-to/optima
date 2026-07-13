"""Shell-free host authority for reproducible release images and containers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping

from optima.release import (
    ContainerReproducibility, PublishedRelease, ReleaseError,
    SignedContainerReproducibility, _stable_regular as _regular_bytes,
    _container_deployment, _container_dockerfile, reopen_release,
    sign_container_reproducibility, verify_container_reproducibility,
)
from optima.stack_identity import canonical_json_bytes, require_sha256_hex
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}\Z")
_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}\Z")
_REGISTRY_HOST = re.compile(r"(?:localhost|[a-z0-9]+(?:[.-][a-z0-9]+)*)(?::[1-9][0-9]{0,4})?\Z")
_REPOSITORY_PATH = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*\Z")
_MANIFEST_TYPES = frozenset({"application/vnd.oci.image.manifest.v1+json", "application/vnd.docker.distribution.manifest.v2+json"})
_CONFIG_TYPES = frozenset({"application/vnd.oci.image.config.v1+json", "application/vnd.docker.container.image.v1+json"})
_LAYER_TYPES = frozenset({
    "application/vnd.oci.image.layer.v1.tar", "application/vnd.oci.image.layer.v1.tar+gzip",
    "application/vnd.oci.image.layer.v1.tar+zstd", "application/vnd.docker.image.rootfs.diff.tar",
    "application/vnd.docker.image.rootfs.diff.tar.gzip",
})
_DESCRIPTOR_LABEL = "org.optima.release.descriptor"
_SECCOMP_LABEL = "org.optima.seccomp.sha256"
_OVERLAY_LABEL = "org.optima.runtime-overlays"
_MANIFEST_LIMIT = 4 << 20
_CONFIG_LIMIT = 16 << 20
_TMPFS_SIZE = 16 << 30
_SHM_SIZE = 32 << 30
_TMPFS_OPTIONS = frozenset({"rw", "nosuid", "nodev", "noexec", f"size={_TMPFS_SIZE}", "mode=1777"})
_DOCKER = "/usr/local/bin/docker"
_RECEIPT_DIR = "/tmp/optima-release-receipts"
class ReleaseHostError(ReleaseError):
    """A registry object or host lifecycle action violated release policy."""
def _digest(value: object, label: str) -> str:
    try:
        result = require_sha256_hex(value, field=label)
    except ValueError as exc:
        raise ReleaseHostError(str(exc)) from None
    if result == "0" * 64:
        raise ReleaseHostError(f"{label} must not be the all-zero digest")
    return result
def _json(raw: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ReleaseHostError(f"{label} contains duplicate JSON keys")
            result[key] = value
        return result
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except ReleaseHostError:
        raise
    except (UnicodeError, ValueError) as exc:
        raise ReleaseHostError(f"{label} is not canonical JSON data: {exc}") from None
def _repository(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or any(char in value for char in "@\x00\r\n"):
        raise ReleaseHostError("registry repository is malformed")
    host, separator, path = value.partition("/")
    if (
        not separator
        or not host
        or host.lower() != host
        or ("." not in host and ":" not in host and host != "localhost")
        or _REGISTRY_HOST.fullmatch(host) is None
        or _REPOSITORY_PATH.fullmatch(path) is None
    ):
        raise ReleaseHostError("registry repository must include an explicit host and path")
    return host, path
def _descriptor(value: object, *, media_types: frozenset[str], label: str) -> "OCIDescriptor":
    if type(value) is not dict or set(value) != {"digest", "mediaType", "size"}:
        raise ReleaseHostError(f"{label} descriptor fields differ")
    digest = value["digest"]
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise ReleaseHostError(f"{label} descriptor digest is malformed")
    media_type = value["mediaType"]
    size = value["size"]
    if media_type not in media_types or type(size) is not int or size < 0:
        raise ReleaseHostError(f"{label} descriptor is unsupported")
    return OCIDescriptor(media_type, _digest(digest[7:], f"{label} digest"), size)
@dataclass(frozen=True)
class OCIDescriptor:
    media_type: str
    sha256: str
    size: int
@dataclass(frozen=True)
class RegistryImage:
    repository: str
    reference: str
    manifest_digest: str
    manifest_media_type: str
    config: OCIDescriptor
    layers: tuple[OCIDescriptor, ...]
    platform: str
    labels: Mapping[str, str]
    entrypoint: tuple[str, ...] | None
    command: tuple[str, ...] | None
    environment: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", MappingProxyType(dict(self.labels)))
    @property
    def digest_reference(self) -> str:
        return f"{self.repository}@sha256:{self.manifest_digest}"

class RegistryV2Client:
    """Minimal raw Registry-v2 reader; it never invokes a Docker client."""
    def __init__(self, repository: str, *, base_url: str | None = None,
                 opener: Callable[..., Any] | None = None) -> None:
        host, path = _repository(repository)
        if base_url is None:
            loopback = host == "localhost" or host.startswith("127.")
            base_url = ("http://" if loopback else "https://") + host
        parsed = urllib.parse.urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.netloc != host
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
            or (parsed.scheme == "http" and not (
                host == "localhost" or host.startswith("127.")
            ))
        ):
            raise ReleaseHostError("registry base URL is not an exact secure origin")
        self.repository = repository
        self._path = urllib.parse.quote(path, safe="/")
        self._origin = f"{parsed.scheme}://{parsed.netloc}"
        self._opener = opener or urllib.request.urlopen
    def _get(self, suffix: str, *, accept: str, limit: int) -> tuple[bytes, Mapping[str, str]]:
        url = f"{self._origin}/v2/{self._path}/{suffix}"
        request = urllib.request.Request(url, headers={"Accept": accept}, method="GET")
        try:
            response = self._opener(request, timeout=30)
            with response:
                final = urllib.parse.urlsplit(response.geturl())
                if f"{final.scheme}://{final.netloc}" != self._origin:
                    raise ReleaseHostError("registry redirected outside its trusted origin")
                status = getattr(response, "status", response.getcode())
                if status != 200:
                    raise ReleaseHostError(f"registry returned HTTP {status}")
                raw = response.read(limit + 1)
                if len(raw) > limit:
                    raise ReleaseHostError("registry object exceeds its size limit")
                headers = {key.lower(): value.strip() for key, value in response.headers.items()}
        except ReleaseHostError:
            raise
        except (OSError, urllib.error.URLError) as exc:
            raise ReleaseHostError(f"registry read failed: {exc}") from None
        return raw, MappingProxyType(headers)
    def reopen(self, reference: str, *, expected_platform: str) -> RegistryImage:
        if not isinstance(reference, str) or (
            _TAG.fullmatch(reference) is None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", reference) is None
        ):
            raise ReleaseHostError("registry tag or digest reference is malformed")
        accepts = ", ".join(sorted(_MANIFEST_TYPES))
        manifest_raw, headers = self._get(
            "manifests/" + urllib.parse.quote(reference, safe=":"),
            accept=accepts,
            limit=_MANIFEST_LIMIT,
        )
        actual = hashlib.sha256(manifest_raw).hexdigest()
        header_digest = headers.get("docker-content-digest", "")
        if header_digest != f"sha256:{actual}":
            raise ReleaseHostError("registry manifest digest header differs from its bytes")
        manifest = _json(manifest_raw, "registry manifest")
        if type(manifest) is not dict or set(manifest) != {
            "schemaVersion", "mediaType", "config", "layers"
        }:
            raise ReleaseHostError("registry manifest fields differ")
        if manifest["schemaVersion"] != 2 or manifest["mediaType"] not in _MANIFEST_TYPES:
            raise ReleaseHostError("registry manifest schema or media type is unsupported")
        config = _descriptor(manifest["config"], media_types=_CONFIG_TYPES, label="config")
        if type(manifest["layers"]) is not list or not manifest["layers"]:
            raise ReleaseHostError("registry manifest has no image layers")
        layers = tuple(
            _descriptor(row, media_types=_LAYER_TYPES, label=f"layer {index}")
            for index, row in enumerate(manifest["layers"])
        )
        config_raw, _ = self._get(
            f"blobs/sha256:{config.sha256}", accept=config.media_type, limit=_CONFIG_LIMIT
        )
        if len(config_raw) != config.size or hashlib.sha256(config_raw).hexdigest() != config.sha256:
            raise ReleaseHostError("registry config bytes differ from their descriptor")
        config_json = _json(config_raw, "registry image config")
        if type(config_json) is not dict:
            raise ReleaseHostError("registry image config is not an object")
        os_name = config_json.get("os")
        architecture = config_json.get("architecture")
        if (
            not isinstance(os_name, str)
            or not isinstance(architecture, str)
            or config_json.get("variant") not in {None, ""}
        ):
            raise ReleaseHostError("registry image platform is ambiguous")
        platform = f"{os_name}/{architecture}"
        if not isinstance(expected_platform, str) or platform != expected_platform or os_name != "linux":
            raise ReleaseHostError("registry image platform differs from the signed release")
        rootfs = config_json.get("rootfs")
        if type(rootfs) is not dict or rootfs.get("type") != "layers" or type(rootfs.get("diff_ids")) is not list:
            raise ReleaseHostError("registry image rootfs identity is malformed")
        diff_ids = rootfs["diff_ids"]
        if len(diff_ids) != len(layers) or any(
            not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
            for value in diff_ids
        ):
            raise ReleaseHostError("registry layer and rootfs descriptor coverage differs")
        runtime = config_json.get("config")
        if type(runtime) is not dict:
            raise ReleaseHostError("registry runtime config is malformed")
        labels = runtime.get("Labels") or {}
        if type(labels) is not dict or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in labels.items()
        ):
            raise ReleaseHostError("registry image labels are malformed")
        entrypoint = _string_array(runtime.get("Entrypoint"), "registry Entrypoint", optional=True)
        command = _string_array(runtime.get("Cmd"), "registry Cmd", optional=True)
        environment = _string_array(runtime.get("Env") or [], "registry Env", optional=False)
        return RegistryImage(
            self.repository,
            reference,
            actual,
            manifest["mediaType"],
            config,
            layers,
            platform,
            labels,
            entrypoint,
            command,
            environment or (),
        )

def _string_array(value: object, label: str, *, optional: bool) -> tuple[str, ...] | None:
    if value is None and optional:
        return None
    if type(value) is not list or any(
        not isinstance(item, str) or "\x00" in item for item in value
    ):
        raise ReleaseHostError(f"{label} is malformed")
    return tuple(value)
def _overlay_digest(release: PublishedRelease) -> str:
    from optima.release_runtime import reviewed_runtime_overlay_digest
    return reviewed_runtime_overlay_digest(
        expected_sglang_version=release.descriptor.sglang_version,
        expected_upstream_revision=release.descriptor.upstream_revision,
    )
def _reopen_context(
    root: str | Path, *, expected_descriptor_digest: str, expected_public_key: bytes | str
) -> tuple[Path, PublishedRelease]:
    requested = Path(root)
    if requested.is_symlink():
        raise ReleaseHostError("container context must not be a symlink")
    path = requested.resolve(strict=True)
    if not path.is_dir() or {item.name for item in path.iterdir()} != {
        "Dockerfile", "deployment.json", "release", "seccomp.json", "trusted-release-key"
    }:
        raise ReleaseHostError("container context inventory differs")
    release = reopen_release(
        path / "release",
        expected_descriptor_digest=expected_descriptor_digest,
        expected_public_key=expected_public_key,
    )
    expected_key = expected_public_key.hex() if type(expected_public_key) is bytes else expected_public_key
    if _regular_bytes(path / "trusted-release-key") != (expected_key + "\n").encode("ascii"):
        raise ReleaseHostError("container context trusted key differs")
    seccomp = _regular_bytes(path / "seccomp.json")
    release_seccomp = _regular_bytes(
        release.root / "artifacts" / release.descriptor.seccomp.name
    )
    if seccomp != release_seccomp or hashlib.sha256(seccomp).hexdigest() != release.descriptor.seccomp.sha256:
        raise ReleaseHostError("container context seccomp bytes differ")
    overlay_digest = _overlay_digest(release)
    deployment = _container_deployment(release, expected_key, overlay_digest)
    if _regular_bytes(path / "deployment.json") != canonical_json_bytes(deployment) + b"\n":
        raise ReleaseHostError("container deployment policy differs")
    if _regular_bytes(path / "Dockerfile") != _container_dockerfile(
        release, overlay_digest
    ):
        raise ReleaseHostError("container Dockerfile differs from the signed release")
    return path, release
def _run(
    argv: tuple[str, ...], *, runner: Callable[..., Any] = subprocess.run, timeout: int
) -> bytes:
    try:
        completed = runner(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReleaseHostError(f"host command failed to execute: {exc}") from None
    if type(completed.returncode) is not int or completed.returncode != 0:
        stderr = bytes(completed.stderr or b"")[-4096:].decode("utf-8", "replace")
        raise ReleaseHostError(f"host command failed ({completed.returncode}): {stderr}")
    return bytes(completed.stdout or b"")
def publish_container_twice(context_root: str | Path, *, repository: str,
    expected_descriptor_digest: str, expected_public_key: bytes | str,
    signing_private_key: bytes, registry_base_url: str | None = None,
    runner: Callable[..., Any] = subprocess.run, opener: Callable[..., Any] | None = None,
) -> SignedContainerReproducibility:
    """Build and publish two no-cache images, then sign their raw common digest."""

    _repository(repository)
    expected = _digest(expected_descriptor_digest, "expected release descriptor")
    context, release = _reopen_context(
        context_root,
        expected_descriptor_digest=expected,
        expected_public_key=expected_public_key,
    )
    platform = release.descriptor.serve.oci_platform
    tags = (f"optima-{expected}-build1", f"optima-{expected}-build2")
    if tags[0] == tags[1] or any(_TAG.fullmatch(tag) is None for tag in tags):
        raise ReleaseHostError("container build tags are not distinct and canonical")
    registry = RegistryV2Client(repository, base_url=registry_base_url, opener=opener)
    images: list[RegistryImage] = []
    for tag in tags:
        image_ref = f"{repository}:{tag}"
        insecure = ",registry.insecure=true" if repository.startswith(("localhost/", "127.")) else ""
        output = (
            f"type=registry,name={image_ref},push=true,oci-mediatypes=true,"
            f"rewrite-timestamp=true{insecure}"
        )
        argv = (
            _DOCKER, "buildx", "build",
            "--builder", "optima-release-builder",
            "--platform", platform,
            "--no-cache", "--pull",
            "--network", "none",
            "--provenance=false", "--sbom=false",
            "--build-arg", "SOURCE_DATE_EPOCH=0",
            "--progress=quiet",
            "--output", output,
            os.fspath(context),
        )
        _run(argv, runner=runner, timeout=7200)
        image = registry.reopen(tag, expected_platform=platform)
        if image.labels.get(_DESCRIPTOR_LABEL) != expected:
            raise ReleaseHostError("published image descriptor label differs")
        if image.labels.get(_SECCOMP_LABEL) != release.descriptor.seccomp.sha256:
            raise ReleaseHostError("published image seccomp label differs")
        if image.labels.get(_OVERLAY_LABEL) != _overlay_digest(release):
            raise ReleaseHostError("published image overlay label differs")
        images.append(image)
    if images[0].manifest_digest != images[1].manifest_digest:
        raise ReleaseHostError("actual Registry-v2 manifests did not reproduce")
    attestation = ContainerReproducibility(
        expected, images[0].manifest_digest, images[1].manifest_digest, platform
    )
    signed = sign_container_reproducibility(attestation, signing_private_key)
    verify_container_reproducibility(
        signed, expected_public_key=expected_public_key
    )
    return signed
@dataclass(frozen=True)
class AuthorizedReleaseContainer:
    release: PublishedRelease
    image: RegistryImage
    repository: str
    descriptor_digest: str
    platform: str
    image_digest: str
    seccomp_path: Path
    @property
    def image_reference(self) -> str:
        return f"{self.repository}@sha256:{self.image_digest}"
def authorize_release_container(release_root: str | Path,
    signed_attestation: SignedContainerReproducibility, registry_image: RegistryImage, *,
    repository: str, expected_public_key: bytes | str, expected_descriptor_digest: str,
    expected_platform: str, expected_image_digest: str,
) -> AuthorizedReleaseContainer:
    """Bind independent host expectations to signed release and registry evidence."""

    expected_descriptor = _digest(expected_descriptor_digest, "expected release descriptor")
    expected_image = _digest(expected_image_digest, "expected release image")
    verify_container_reproducibility(
        signed_attestation, expected_public_key=expected_public_key
    )
    release = reopen_release(
        release_root,
        expected_descriptor_digest=expected_descriptor,
        expected_public_key=expected_public_key,
    )
    attestation = signed_attestation.attestation
    if (
        attestation.descriptor_digest != expected_descriptor
        or attestation.oci_platform != expected_platform
        or release.descriptor.serve.oci_platform != expected_platform
        or attestation.first_oci_digest != expected_image
    ):
        raise ReleaseHostError("container attestation differs from host expectations")
    _repository(repository)
    if (
        not isinstance(registry_image, RegistryImage)
        or registry_image.repository != repository
        or registry_image.manifest_digest != expected_image
        or registry_image.platform != expected_platform
        or registry_image.labels.get(_DESCRIPTOR_LABEL) != expected_descriptor
        or registry_image.labels.get(_SECCOMP_LABEL) != release.descriptor.seccomp.sha256
        or registry_image.labels.get(_OVERLAY_LABEL) != _overlay_digest(release)
    ):
        raise ReleaseHostError("actual registry image differs from signed release authority")
    seccomp_path = (
        release.root / "artifacts" / release.descriptor.seccomp.name
    ).resolve(strict=True)
    seccomp = _regular_bytes(seccomp_path)
    if hashlib.sha256(seccomp).hexdigest() != release.descriptor.seccomp.sha256:
        raise ReleaseHostError("host seccomp profile differs from signed bytes")
    return AuthorizedReleaseContainer(
        release, registry_image, repository, expected_descriptor,
        expected_platform, expected_image, seccomp_path,
    )
def _one_json(raw: bytes, label: str) -> dict[str, Any]:
    value = _json(raw, label)
    if type(value) is not list or len(value) != 1 or type(value[0]) is not dict:
        raise ReleaseHostError(f"{label} did not return exactly one object")
    return value[0]
def _none(value: object) -> bool:
    return value is None or value == [] or value == {} or value == ""
def _verify_local_image(row: dict[str, Any], authorization: AuthorizedReleaseContainer) -> None:
    config = row.get("Config")
    if type(config) is not dict:
        raise ReleaseHostError("local image inspect lacks Config")
    image = authorization.image
    os_name, architecture = authorization.platform.split("/", 1)
    if (
        row.get("Id") != f"sha256:{image.config.sha256}"
        or authorization.image_reference not in (row.get("RepoDigests") or [])
        or row.get("Os") != os_name
        or row.get("Architecture") != architecture
        or config.get("Entrypoint") != (list(image.entrypoint) if image.entrypoint is not None else None)
        or config.get("Cmd") != (list(image.command) if image.command is not None else None)
        or tuple(config.get("Env") or ()) != image.environment
        or config.get("Volumes") not in (None, {})
    ):
        raise ReleaseHostError("pulled image inspect differs from registry identity")
    labels = config.get("Labels") or {}
    if (
        type(labels) is not dict
        or labels.get(_DESCRIPTOR_LABEL) != authorization.descriptor_digest
        or labels.get(_SECCOMP_LABEL) != authorization.release.descriptor.seccomp.sha256
        or labels.get(_OVERLAY_LABEL) != _overlay_digest(authorization.release)
    ):
        raise ReleaseHostError("pulled image labels differ")
def _verify_device_request(value: object) -> bool:
    return type(value) is list and len(value) == 1 and type(value[0]) is dict and (
        value[0].get("Count") == -1 and value[0].get("DeviceIDs") in (None, [])
        and value[0].get("Capabilities") == [["gpu"]]
        and value[0].get("Options") in (None, {}) and value[0].get("Driver") in (None, "")
    )
def _verify_host_model_mount(host: Mapping[str, Any], expected_bind: str, model_root: Path,
                             destination: str) -> bool:
    binds = host.get("Binds")
    mounts = host.get("Mounts")
    via_bind = binds == [expected_bind] and _none(mounts)
    via_mount = _none(binds) and type(mounts) is list and len(mounts) == 1
    if via_mount:
        row = mounts[0]
        via_mount = type(row) is dict and (
            row.get("Type") == "bind" and row.get("Source") == os.fspath(model_root)
            and row.get("Target") == destination and row.get("ReadOnly") is True
            and row.get("Consistency", "") in {"", "default"}
        )
    return via_bind or via_mount
def _verify_container_inspect(row: dict[str, Any], authorization: AuthorizedReleaseContainer,
    *, cid: str, name: str, model_root: Path) -> None:
    config, host, state, mounts = (
        row.get("Config"), row.get("HostConfig"), row.get("State"), row.get("Mounts")
    )
    destination = authorization.release.descriptor.serve.model_mount
    expected_bind = f"{model_root}:{destination}:ro"
    if (
        row.get("Id") != cid
        or row.get("Name") != f"/{name}"
        or row.get("Image") != f"sha256:{authorization.image.config.sha256}"
        or type(config) is not dict
        or config.get("Image") != authorization.image_reference
        or config.get("Entrypoint") != (list(authorization.image.entrypoint) if authorization.image.entrypoint is not None else None)
        or config.get("Cmd") != (list(authorization.image.command) if authorization.image.command is not None else None)
        or config.get("Env") != list(authorization.image.environment)
        or config.get("Labels") != dict(authorization.image.labels)
        or type(state) is not dict
        or state.get("Status") != "created"
        or state.get("Running") is not False
        or type(host) is not dict
        or host.get("ReadonlyRootfs") is not True
        or host.get("Privileged") is not False
        or host.get("NetworkMode") != "host"
        or host.get("IpcMode") != "host"
        or host.get("ShmSize") != _SHM_SIZE
        or host.get("CapDrop") != ["ALL"]
        or not _none(host.get("CapAdd"))
        or not _verify_host_model_mount(host, expected_bind, model_root, destination)
        or not _verify_device_request(host.get("DeviceRequests"))
        or set(host.get("SecurityOpt") or ()) != {
            "no-new-privileges=true", f"seccomp={authorization.seccomp_path}"
        }
        or set((host.get("Tmpfs") or {}).keys()) != {"/tmp"}
        or frozenset((host.get("Tmpfs") or {}).get("/tmp", "").split(",")) != _TMPFS_OPTIONS
        or type(mounts) is not list or len(mounts) != 1
    ):
        raise ReleaseHostError("created container violates the closed host policy")
    mount = mounts[0]
    if type(mount) is not dict or (
        mount.get("Type") != "bind" or mount.get("Source") != os.fspath(model_root)
        or mount.get("Destination") != destination or mount.get("RW") is not False
    ):
        raise ReleaseHostError("created container model mount differs")
class ReleaseContainerHandle:
    """A created digest-pinned container; every start rechecks its policy."""
    def __init__(self, authorization: AuthorizedReleaseContainer, cid: str, name: str,
                 model_root: Path, runner: Callable[..., Any]) -> None:
        self.authorization = authorization
        self.cid = cid
        self.name = name
        self.model_root = model_root
        self._runner = runner
        self._destroyed = False
    def inspect_before_start(self) -> None:
        if self._destroyed:
            raise ReleaseHostError("release container was destroyed")
        raw = _run(
            (_DOCKER, "container", "inspect", self.cid),
            runner=self._runner, timeout=30,
        )
        _verify_container_inspect(
            _one_json(raw, "container inspect"), self.authorization,
            cid=self.cid, name=self.name, model_root=self.model_root,
        )
    def start(self) -> None:
        self.inspect_before_start()
        _run(
            (_DOCKER, "container", "start", self.cid),
            runner=self._runner, timeout=60,
        )
    def copy_receipts(self, destination: str | Path) -> Path:
        if self._destroyed:
            raise ReleaseHostError("release container was destroyed")
        target = Path(destination)
        if target.exists() or target.is_symlink():
            raise ReleaseHostError("receipt destination already exists")
        target.parent.resolve(strict=True)
        target.mkdir(mode=0o700)
        try:
            _run(
                (_DOCKER, "container", "cp", f"{self.cid}:{_RECEIPT_DIR}/.", os.fspath(target.resolve())),
                runner=self._runner, timeout=120,
            )
        except BaseException:
            try:
                target.rmdir()
            except OSError:
                pass
            raise
        return target
    def destroy(self) -> None:
        if self._destroyed:
            return
        _run(
            (_DOCKER, "container", "rm", "--force", "--volumes", self.cid),
            runner=self._runner, timeout=120,
        )
        self._destroyed = True
def create_release_container(authorization: AuthorizedReleaseContainer, model_root: str | Path,
    *, name: str, runner: Callable[..., Any] = subprocess.run) -> ReleaseContainerHandle:
    """Pull by digest and create, but do not start, one closed serving container."""

    if not isinstance(authorization, AuthorizedReleaseContainer):
        raise ReleaseHostError("container authorization is untyped")
    if not isinstance(name, str) or _NAME.fullmatch(name) is None:
        raise ReleaseHostError("container name is malformed")
    requested_model = Path(model_root)
    if requested_model.is_symlink():
        raise ReleaseHostError("model root must not be a symlink")
    model = requested_model.resolve(strict=True)
    if not model.is_dir() or model == Path("/") or any(
        char in os.fspath(model) for char in ",\x00\r\n"
    ):
        raise ReleaseHostError("model root is not one exact directory")
    image_ref = authorization.image_reference
    _run(
        (_DOCKER, "image", "pull", "--platform", authorization.platform, image_ref),
        runner=runner, timeout=7200,
    )
    image_row = _one_json(
        _run((_DOCKER, "image", "inspect", image_ref), runner=runner, timeout=30),
        "image inspect",
    )
    _verify_local_image(image_row, authorization)
    tmpfs = f"/tmp:{','.join(sorted(_TMPFS_OPTIONS))}"
    mount = (
        f"type=bind,src={model},dst={authorization.release.descriptor.serve.model_mount},readonly"
    )
    argv = (
        _DOCKER, "container", "create",
        "--name", name,
        "--platform", authorization.platform,
        "--read-only",
        "--tmpfs", tmpfs,
        "--security-opt", "no-new-privileges=true",
        "--security-opt", f"seccomp={authorization.seccomp_path}",
        "--cap-drop", "ALL",
        "--mount", mount,
        "--gpus", "all",
        "--ipc", "host",
        "--shm-size", str(_SHM_SIZE),
        "--network", "host",
        image_ref,
    )
    cid = _run(argv, runner=runner, timeout=120).decode("ascii", "strict").strip()
    if re.fullmatch(r"[0-9a-f]{64}", cid) is None:
        raise ReleaseHostError("Docker create returned a malformed container ID")
    handle = ReleaseContainerHandle(authorization, cid, name, model, runner)
    try:
        handle.inspect_before_start()
    except BaseException:
        try:
            _run(
                (_DOCKER, "container", "rm", "--force", "--volumes", cid),
                runner=runner, timeout=120,
            )
        except BaseException:
            pass
        raise
    return handle
__all__ = [
    "AuthorizedReleaseContainer", "OCIDescriptor", "RegistryImage",
    "RegistryV2Client", "ReleaseContainerHandle", "ReleaseHostError",
    "authorize_release_container", "create_release_container",
    "publish_container_twice",
]
