"""Fail-closed entrypoint for a signed Optima Engine serving release.

The wheel bootstrap sees no active candidate in this first interpreter.  Only after
the signed release, bundled native publication, complete model receipt, deployment
seccomp state, environment, and exact SGLang command reopen does this module exec a
fresh Python interpreter with the candidate seams armed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path


class ReleaseRuntimeError(RuntimeError):
    """A serving release cannot safely enter the engine process."""


@dataclass(frozen=True)
class VerifiedServingRelease:
    descriptor_digest: str
    command: tuple[str, ...]
    environment: tuple[tuple[str, str], ...]


_RECEIPT_ROOT = Path("/tmp/optima-release-receipts")
_PYTHON = "/usr/bin/python3"
_OVERLAY_ROOT = Path("/sgl-workspace/sglang")
_BASE_ENVIRONMENT = {
    "HOME": "/tmp",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin",
    "TMPDIR": "/tmp",
}
_GPU_ENVIRONMENT = ("NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_VISIBLE_DEVICES")


_STABLE_STAT_FIELDS = (
    "st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns",
)


def _read_public_key(path: str | os.PathLike[str]) -> str:
    value = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(value, flags)
    except OSError as exc:
        raise ReleaseRuntimeError(f"cannot open trusted release key: {exc}") from None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > 256:
            raise ReleaseRuntimeError("trusted release key is not one bounded regular file")
        raw = os.read(fd, before.st_size + 1)
        after = os.fstat(fd)
        if (
            any(getattr(before, field) != getattr(after, field) for field in _STABLE_STAT_FIELDS)
            or len(raw) != before.st_size
        ):
            raise ReleaseRuntimeError("trusted release key changed while reading")
    finally:
        os.close(fd)
    try:
        key = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        raise ReleaseRuntimeError("trusted release key is not ASCII") from None
    if re.fullmatch(r"[0-9a-f]{64}", key) is None:
        raise ReleaseRuntimeError("trusted release key is not raw Ed25519 hex")
    return key


def _seccomp_active(status_path: str | os.PathLike[str] = "/proc/self/status") -> bool:
    try:
        rows = Path(status_path).read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError):
        return False
    values = [row.split(":", 1)[1].strip() for row in rows if row.startswith("Seccomp:")]
    return values == ["2"]


def _regular_bytes(path: Path, *, limit: int = 16 << 20) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReleaseRuntimeError(f"reviewed runtime overlay is unavailable: {exc}") from None
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or info.st_size < 1
        or info.st_size > limit
    ):
        raise ReleaseRuntimeError("reviewed runtime overlay is not one bounded regular file")
    payload = path.read_bytes()
    if len(payload) != info.st_size:
        raise ReleaseRuntimeError("reviewed runtime overlay changed while reading")
    return payload


def _reviewed_runtime_overlay_inventory(
    *, expected_sglang_version: str, expected_upstream_revision: str
) -> tuple[tuple[str, str, bytes, dict[str, object]], ...]:
    package_root = Path(__file__).resolve().parent.parent
    provenance_path = Path(__file__).resolve().parent / "vendor_provenance.json"
    try:
        provenance = json.loads(_regular_bytes(provenance_path))
        assets = provenance["assets"]
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise ReleaseRuntimeError(f"vendor provenance is malformed: {exc}") from None
    rows = [row for row in assets if isinstance(row, dict) and "runtime_target" in row]
    if len(rows) != 2:
        raise ReleaseRuntimeError("reviewed runtime overlay inventory differs")
    inventory: list[tuple[str, str, bytes, dict[str, object]]] = []
    for row in rows:
        packaged = row.get("packaged")
        target = row.get("runtime_target")
        if not isinstance(packaged, dict) or not isinstance(target, dict):
            raise ReleaseRuntimeError("reviewed runtime overlay metadata is malformed")
        if (
            target.get("package_version") != expected_sglang_version
            or target.get("revision") != expected_upstream_revision
        ):
            raise ReleaseRuntimeError("reviewed runtime overlay targets another SGLang pin")
        source_name = packaged.get("path")
        target_name = target.get("path")
        if (
            not isinstance(source_name, str)
            or not source_name.startswith("optima/arena_assets/minimax_m3/sglang_patch/")
            or not isinstance(target_name, str)
            or not target_name.startswith("python/sglang/")
            or ".." in Path(source_name).parts
            or ".." in Path(target_name).parts
        ):
            raise ReleaseRuntimeError("reviewed runtime overlay path is outside its policy")
        source = package_root / source_name
        source_payload = _regular_bytes(source)
        if (
            len(source_payload) != packaged.get("size")
            or hashlib.sha256(source_payload).hexdigest() != packaged.get("sha256")
        ):
            raise ReleaseRuntimeError("reviewed runtime overlay source changed")
        inventory.append((source_name, target_name, source_payload, target))
    return tuple(sorted(inventory))


def reviewed_runtime_overlay_digest(
    *, expected_sglang_version: str, expected_upstream_revision: str
) -> str:
    identity = [
        {
            "source": source_name,
            "target": target_name,
            "sha256": hashlib.sha256(source_payload).hexdigest(),
            "size": len(source_payload),
        }
        for source_name, target_name, source_payload, _target in (
            _reviewed_runtime_overlay_inventory(
                expected_sglang_version=expected_sglang_version,
                expected_upstream_revision=expected_upstream_revision,
            )
        )
    ]
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def install_reviewed_runtime_overlays(
    *, expected_sglang_version: str, expected_upstream_revision: str
) -> str:
    """Install the two reviewed MiniMax compatibility files during image build."""

    inventory = _reviewed_runtime_overlay_inventory(
        expected_sglang_version=expected_sglang_version,
        expected_upstream_revision=expected_upstream_revision,
    )
    for _source_name, target_name, source_payload, target in inventory:
        destination = _OVERLAY_ROOT / target_name
        stock_payload = _regular_bytes(destination)
        if (
            len(stock_payload) != target.get("size")
            or hashlib.sha256(stock_payload).hexdigest() != target.get("sha256")
        ):
            raise ReleaseRuntimeError("reviewed runtime overlay stock target changed")
        temporary = destination.with_name(f".{destination.name}.optima-overlay")
        try:
            fd = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                stat.S_IMODE(destination.stat().st_mode),
            )
            try:
                view = memoryview(source_payload)
                while view:
                    written = os.write(fd, view)
                    if written < 1:
                        raise OSError("short runtime overlay write")
                    view = view[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(temporary, destination)
        except OSError as exc:
            try:
                temporary.unlink()
            except OSError:
                pass
            raise ReleaseRuntimeError(f"reviewed runtime overlay install failed: {exc}") from None
    return reviewed_runtime_overlay_digest(
        expected_sglang_version=expected_sglang_version,
        expected_upstream_revision=expected_upstream_revision,
    )


def verify_serving_release(
    *,
    release_root: str | os.PathLike[str],
    expected_public_key: str,
    model_root: str | os.PathLike[str],
    command: tuple[str, ...],
    require_seccomp: bool,
) -> VerifiedServingRelease:
    """Reopen every serving input without importing or loading candidate code."""

    from optima.manifest import load_manifest
    from optima.model_provision import reopen_embedded_model_provision
    from optima.release import ReleaseError, reopen_release
    from optima.seams import SEAM_ADAPTERS, seam_binding_environment

    try:
        release = reopen_release(
            release_root,
            expected_public_key=expected_public_key,
        )
    except ReleaseError as exc:
        raise ReleaseRuntimeError(f"release verification failed: {exc}") from None
    descriptor = release.descriptor
    expected_command = (
        _PYTHON, "-m", "sglang.launch_server",
        *descriptor.serve.command_arguments,
    )
    if tuple(command) != expected_command:
        raise ReleaseRuntimeError("serving command differs from the signed ServeSpec")
    if Path(model_root).as_posix() != descriptor.serve.model_mount:
        raise ReleaseRuntimeError("mounted model path differs from the signed ServeSpec")
    receipt_path = release.root / "artifacts" / descriptor.model_receipt.name
    try:
        reopen_embedded_model_provision(
            model_root,
            receipt_path,
            expected_content_digest=descriptor.model.content_digest,
            expected_receipt_digest=descriptor.model.receipt_digest,
        )
    except Exception as exc:
        raise ReleaseRuntimeError(f"model verification failed: {exc}") from None
    if require_seccomp and not _seccomp_active():
        raise ReleaseRuntimeError("serving process is not running under seccomp filter mode")
    for key, value in descriptor.serve.environment:
        if os.environ.get(key) != value:
            raise ReleaseRuntimeError(f"serving environment differs for signed key {key}")

    try:
        manifest = load_manifest(release.root / "engine-tree")
        active_slots = {operation.slot for operation in manifest.ops}
        seam_bindings = tuple(
            sorted(
                {
                    adapter.binding_id
                    for adapter in SEAM_ADAPTERS
                    if adapter.binding_id is not None
                    and active_slots.intersection(adapter.slots)
                }
            )
        )
        gate_environment = seam_binding_environment(seam_bindings)
    except Exception as exc:
        raise ReleaseRuntimeError(f"release seam binding failed: {exc}") from None

    native_root = (
        release.root / "native-artifacts" / descriptor.native.build_spec_digest[:2]
        / descriptor.native.build_spec_digest
    )
    environment = {
        **dict(descriptor.serve.environment),
        "OPTIMA_ACTIVE": "1",
        "OPTIMA_BUNDLE_PATH": "/optima/engine-tree",
        "OPTIMA_ENGINE_TREE_DIGEST": descriptor.engine_tree_digest,
        "OPTIMA_ENGINE_WORKER": "1",
        "OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST": descriptor.native.publication_digest,
        "OPTIMA_NATIVE_ARTIFACT_ROOT": str(native_root),
        "OPTIMA_NATIVE_BUILD_SPEC_DIGEST": descriptor.native.build_spec_digest,
        "OPTIMA_PREBUILT_ARTIFACTS": "1",
        "OPTIMA_REBUILD_PHASE": "load",
        "OPTIMA_RELEASE_DESCRIPTOR_DIGEST": descriptor.digest,
        "OPTIMA_RELEASE_REQUIRED": "1",
        "OPTIMA_RELEASE_VERIFIED": descriptor.digest,
        "OPTIMA_STACK_DIGEST": descriptor.release_manifest.digest,
        "OPTIMA_STRICT": "1",
        "OPTIMA_TARGET_GPU_ARCH": descriptor.native.build_spec["target_architecture"],
        "OPTIMA_MODEL_CONTENT_DIGEST": descriptor.model.content_digest,
        "OPTIMA_MODEL_MANIFEST_DIGEST": descriptor.model.manifest_digest,
        "OPTIMA_RUNTIME_DIGEST": descriptor.release_manifest.runtime_digest,
        "OPTIMA_WORKER_DISTRIBUTION_DIGEST": descriptor.native.build_spec["worker_distribution_digest"],
        "OPTIMA_EXPECTED_SGLANG_VERSION": descriptor.sglang_version,
        "OPTIMA_SEAM_RECEIPT_DIR": str(_RECEIPT_ROOT),
        "SGLANG_PLUGINS": "optima",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        **gate_environment,
    }
    return VerifiedServingRelease(
        descriptor.digest,
        expected_command,
        tuple(sorted(environment.items())),
    )


def _closed_serving_environment(
    verified: VerifiedServingRelease, ambient: dict[str, str]
) -> dict[str, str]:
    environment = dict(_BASE_ENVIRONMENT)
    for name in _GPU_ENVIRONMENT:
        value = ambient.get(name)
        if value is not None:
            if not value or len(value) > 4096 or any(char in value for char in "\x00\r\n"):
                raise ReleaseRuntimeError(f"runtime GPU environment {name} is malformed")
            environment[name] = value
    environment.update(dict(verified.environment))
    return environment


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="verify and enter a signed Optima Engine release")
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--expected-public-key")
    parser.add_argument("--expected-public-key-file")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--require-seccomp", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv[:1] == ["install-reviewed-overlays"]:
        overlay_parser = argparse.ArgumentParser()
        overlay_parser.add_argument("_command", choices=("install-reviewed-overlays",))
        overlay_parser.add_argument("--expected-sglang-version", required=True)
        overlay_parser.add_argument("--expected-upstream-revision", required=True)
        overlay_parser.add_argument("--expected-overlay-digest", required=True)
        overlay_args = overlay_parser.parse_args(raw_argv)
        digest = install_reviewed_runtime_overlays(
            expected_sglang_version=overlay_args.expected_sglang_version,
            expected_upstream_revision=overlay_args.expected_upstream_revision,
        )
        if digest != overlay_args.expected_overlay_digest:
            raise ReleaseRuntimeError("reviewed runtime overlay identity differs")
        print(digest)
        return 0
    args = _parser().parse_args(raw_argv)
    if bool(args.expected_public_key) == bool(args.expected_public_key_file):
        raise ReleaseRuntimeError(
            "exactly one trusted expected public key source is required"
        )
    key = (
        args.expected_public_key
        if args.expected_public_key is not None
        else _read_public_key(args.expected_public_key_file)
    )
    command = tuple(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if Path(args.release_root).resolve() != Path("/optima"):
        raise ReleaseRuntimeError("production release root must be /optima")
    verified = verify_serving_release(
        release_root=args.release_root,
        expected_public_key=key,
        model_root=args.model_root,
        command=command,
        require_seccomp=args.require_seccomp,
    )
    try:
        if _RECEIPT_ROOT.is_symlink():
            raise ReleaseRuntimeError("fixed serve receipt root must not be a symlink")
        _RECEIPT_ROOT.mkdir(mode=0o700)
    except FileExistsError:
        if not _RECEIPT_ROOT.is_dir() or any(_RECEIPT_ROOT.iterdir()):
            raise ReleaseRuntimeError("fixed serve receipt root is not fresh") from None
    except OSError as exc:
        raise ReleaseRuntimeError(f"cannot create fixed serve receipt root: {exc}") from None
    environment = _closed_serving_environment(verified, dict(os.environ))
    os.execve(command[0], command, environment)
    return 127


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseRuntimeError as exc:
        print(f"optima release refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


__all__ = [
    "ReleaseRuntimeError", "VerifiedServingRelease", "install_reviewed_runtime_overlays",
    "main", "reviewed_runtime_overlay_digest", "verify_serving_release",
]
