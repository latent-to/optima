"""Fail-closed entrypoint for a signed Optima Engine serving release.

The wheel bootstrap sees no active candidate in this first interpreter.  Only after
the signed release, bundled native publication, complete model receipt, deployment
seccomp state, environment, and exact SGLang command reopen does this module exec a
fresh Python interpreter with the candidate seams armed.
"""

from __future__ import annotations

import argparse
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


def verify_serving_release(
    *,
    release_root: str | os.PathLike[str],
    expected_public_key: str,
    model_root: str | os.PathLike[str],
    command: tuple[str, ...],
    require_seccomp: bool,
) -> VerifiedServingRelease:
    """Reopen every serving input without importing or loading candidate code."""

    from optima.model_provision import reopen_embedded_model_provision
    from optima.release import ReleaseError, reopen_release

    try:
        release = reopen_release(
            release_root,
            expected_public_key=expected_public_key,
        )
    except ReleaseError as exc:
        raise ReleaseRuntimeError(f"release verification failed: {exc}") from None
    descriptor = release.descriptor
    expected_command = (
        "python", "-m", "sglang.launch_server",
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

    native_root = (
        release.root / "native-artifacts" / descriptor.native.build_spec_digest[:2]
        / descriptor.native.build_spec_digest
    )
    environment = {
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
    }
    return VerifiedServingRelease(
        descriptor.digest,
        expected_command,
        tuple(sorted(environment.items())),
    )


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
    args = _parser().parse_args(argv)
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
    environment = dict(os.environ)
    for name in tuple(environment):
        if (
            name.startswith(("OPTIMA_", "PYTHON", "LD_", "DYLD_"))
            or name in {"SGLANG_PLUGINS", "SGLANG_PLUGIN_PATH"}
        ):
            del environment[name]
    environment.update(dict(verified.environment))
    os.execvpe(command[0], command, environment)
    return 127


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReleaseRuntimeError as exc:
        print(f"optima release refused: {exc}", file=sys.stderr)
        raise SystemExit(2) from None


__all__ = [
    "ReleaseRuntimeError", "VerifiedServingRelease", "main",
    "verify_serving_release",
]
