"""Build or load inspectable CUDA extensions without crossing the host boundary.

``build`` runs only in the disposable, GPU-free rebuild container.  It parses the
materialized manifest as data, compiles only the units declaratively eligible for the
target architecture into a private output stage, validates compiler dependency files
against that selected native set and image-bound roots, and never imports the
resulting native code.  Every declared native source remains in whole-tree identity
and provenance even when its variant is off-domain for this build.

``load`` runs only in an isolated engine worker.  It fully reopens the immutable
host publication, rederives every unit identity from the mounted engine tree, and
then dlopens the exact artifacts.  It creates no directory or lock and cannot
compile, repair, or fall back.  ``all`` is retained solely for direct, non-crownable
development diagnostics.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import uuid
from pathlib import Path
from typing import Any


_SCHEMA = "optima.cuda-extension-set.v3"
_ARTIFACT_DIR = "cuda"
_INDEX_NAME = "extensions.json"
_COMPILE_FLAGS = (
    "-O3",
    "--use_fast_math",
    "--std=c++17",
    "-Xcompiler=-fPIC",
    "-shared",
)
_LINK_LIBRARIES = ("torch", "torch_python", "c10", "c10_cuda", "torch_cuda")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ARCH_RE = re.compile(r"sm(?P<number>[0-9]{2,3})(?P<suffix>[a-z]?)\Z")
_ARCH_SPECIFIC_TARGETS = frozenset({"90", "100", "101", "103", "120", "121"})
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_PHASES = frozenset({"all", "build", "load"})
_INDEX_FIELDS = frozenset(
    {
        "build_context",
        "build_spec_digest",
        "patcher_sha256",
        "schema",
        "selected_sources",
        "source_inventory",
        "target_architecture",
        "tree_digest",
        "units",
    }
)
_UNIT_FIELDS = frozenset(
    {
        "alias",
        "artifact_id",
        "artifact_path",
        "artifact_sha256",
        "dependencies",
        "depfile_path",
        "depfile_sha256",
        "module_name",
        "source",
    }
)
_CONTEXT_FIELDS = frozenset(
    {
        "compiler_env_digest",
        "cxx11_abi",
        "link_libraries",
        "nvcc",
        "nvcc_architecture",
        "nvcc_flags",
        "pinned_build_roots",
        "ptxas",
        "python_include",
        "python_soabi",
        "python_version",
        "torch_api_include",
        "torch_cuda_version",
        "torch_include",
        "torch_lib",
        "torch_version",
    }
)
_TOOL_FIELDS = frozenset({"path", "sha256", "version"})
_SOURCE_FIELDS = frozenset({"path", "sha256"})
_DEPENDENCY_FIELDS = frozenset({"kind", "path", "root", "sha256"})
_MAX_JSON_BYTES = 16 << 20
_MAX_DEPFILE_BYTES = 16 << 20


class CUDAExtensionError(RuntimeError):
    """The declared CUDA product cannot be built or loaded exactly."""


def _log(message: str) -> None:
    print(f"[optima.build_cuda_ext] {message}", flush=True)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise CUDAExtensionError(f"{field} must be a lowercase SHA-256 digest")
    if value == "0" * 64:
        raise CUDAExtensionError(f"{field} must not be the all-zero digest")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CUDAExtensionError(f"native extension JSON repeats key {key!r}")
        result[key] = value
    return result


def _read_json(path: Path, *, maximum: int = _MAX_JSON_BYTES) -> object:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CUDAExtensionError(f"native extension metadata is unavailable: {path}: {exc}") from None
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or before.st_size < 0
        or before.st_size > maximum
    ):
        raise CUDAExtensionError(f"native extension metadata has an unsafe shape: {path}")
    try:
        raw = path.read_bytes()
        after = path.lstat()
        if (
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        ):
            raise CUDAExtensionError(f"native extension metadata changed while reading: {path}")
        return json.loads(raw, object_pairs_hook=_strict_object)
    except CUDAExtensionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CUDAExtensionError(f"native extension metadata is malformed: {path}: {exc}") from None


def _sha256_file(path: Path, *, single_link: bool = True) -> str:
    try:
        before = path.lstat()
    except OSError as exc:
        raise CUDAExtensionError(f"cannot inspect native build input {path}: {exc}") from None
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise CUDAExtensionError(f"native build input is not a regular non-symlink file: {path}")
    if single_link and before.st_nlink != 1:
        raise CUDAExtensionError(f"native build input is hardlinked: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 << 20), b""):
                digest.update(chunk)
        after = path.lstat()
    except OSError as exc:
        raise CUDAExtensionError(f"cannot hash native build input {path}: {exc}") from None
    identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise CUDAExtensionError(f"native build input changed while hashing: {path}")
    return digest.hexdigest()


def _patcher_hash() -> str:
    return _sha256_file(Path(__file__).resolve())


def _phase() -> str:
    phase = os.environ.get("OPTIMA_REBUILD_PHASE", "all").strip().lower()
    if phase not in _PHASES:
        raise CUDAExtensionError(f"unsupported OPTIMA_REBUILD_PHASE: {phase!r}")
    return phase


def _canonical_architecture(value: object, *, required: bool = True) -> str:
    if not isinstance(value, str):
        value = ""
    value = value.strip().lower().replace("_", "")
    match = _ARCH_RE.fullmatch(value)
    if match is None or match.group("suffix") not in {"", "a"} or (
        match.group("suffix") == "a"
        and match.group("number") not in _ARCH_SPECIFIC_TARGETS
    ):
        if required:
            raise CUDAExtensionError(
                "native build requires canonical OPTIMA_TARGET_GPU_ARCH such as sm103 or sm120"
            )
        return ""
    return value


def _nvcc_architecture(architecture: str) -> str:
    match = _ARCH_RE.fullmatch(architecture)
    assert match is not None
    number = match.group("number")
    suffix = match.group("suffix") or ("a" if number in _ARCH_SPECIFIC_TARGETS else "")
    return f"sm_{number}{suffix}"


def _bundle_root() -> Path | None:
    raw = os.environ.get("OPTIMA_BUNDLE_PATH", "").strip()
    if not raw:
        _log("no OPTIMA_BUNDLE_PATH set; nothing to do")
        return None
    requested = Path(raw)
    try:
        if requested.is_symlink():
            raise CUDAExtensionError("materialized engine tree must not be a symlink")
        root = requested.resolve(strict=True)
    except OSError as exc:
        raise CUDAExtensionError(f"materialized engine tree is unavailable: {exc}") from None
    if not root.is_dir():
        raise CUDAExtensionError("materialized engine tree is not a directory")
    return root


def _production_identity(phase: str) -> tuple[str, str, str]:
    build_spec = os.environ.get("OPTIMA_NATIVE_BUILD_SPEC_DIGEST", "").strip()
    tree_digest = os.environ.get("OPTIMA_ENGINE_TREE_DIGEST", "").strip()
    architecture = os.environ.get("OPTIMA_TARGET_GPU_ARCH", "").strip()
    if phase in {"build", "load"}:
        return (
            _require_digest(build_spec, field="OPTIMA_NATIVE_BUILD_SPEC_DIGEST"),
            _require_digest(tree_digest, field="OPTIMA_ENGINE_TREE_DIGEST"),
            _canonical_architecture(architecture),
        )
    return build_spec, tree_digest, _canonical_architecture(architecture, required=False)


def _variant_accepts_architecture(bundle: Path, op, architecture: str) -> bool:
    """Project one variant's declarative eligibility onto build architecture.

    Shape, dtype, graph, and runtime predicates are deliberately ignored here: all
    mutually exclusive variants for the target architecture must be available to the
    live dispatcher.  Legacy and normative architecture predicates retain their
    intersection semantics.  Metadata is parsed as data and candidate Python is never
    imported.
    """

    from optima.registry import eligibility_from_metadata

    metadata = None if op.metadata is None else _read_json(bundle / op.metadata)
    try:
        eligibility = eligibility_from_metadata(
            metadata,
            op.dtypes,
            op.architectures,
        )
    except (TypeError, ValueError) as exc:
        raise CUDAExtensionError(
            f"native variant {op.variant!r} for {op.slot!r} has invalid "
            f"eligibility metadata: {exc}"
        ) from None
    if eligibility.architectures and architecture not in eligibility.architectures:
        return False
    return all(
        predicate.accepts(architecture)
        for predicate in eligibility.capabilities.predicates
        if predicate.field == "architecture"
    )


def _declared_sources(
    bundle: Path,
    architecture: str,
) -> tuple[
    list[tuple[Path, str, str]],
    list[dict[str, str]],
    frozenset[str],
]:
    """Return selected compilation units, full inventory, and selected native set."""

    from optima.manifest import (
        all_declared_cuda_sources,
        load_manifest,
        resolve_cuda_sources,
    )

    manifest = load_manifest(bundle)
    architecture = _canonical_architecture(architecture)
    declared = sorted(
        (Path(path).resolve() for path in all_declared_cuda_sources(bundle, manifest)),
        key=lambda path: path.relative_to(bundle).as_posix(),
    )
    inventory: list[dict[str, str]] = []
    for path in declared:
        try:
            relative = path.relative_to(bundle).as_posix()
        except ValueError:
            raise CUDAExtensionError(
                f"declared CUDA source escapes engine tree: {path}"
            ) from None
        inventory.append({"path": relative, "sha256": _sha256_file(path)})

    selected_paths: set[Path] = set()
    for op in sorted(manifest.ops, key=lambda row: (row.slot, row.variant)):
        if not op.cuda_sources:
            continue
        if _variant_accepts_architecture(bundle, op, architecture):
            selected_paths.update(
                Path(path).resolve() for path in resolve_cuda_sources(bundle, op)
            )
    selected = frozenset(
        path.relative_to(bundle).as_posix() for path in selected_paths
    )

    units: list[tuple[Path, str, str]] = []
    aliases: dict[str, str] = {}
    for path in sorted(
        selected_paths,
        key=lambda row: row.relative_to(bundle).as_posix(),
    ):
        relative = path.relative_to(bundle).as_posix()
        if path.suffix != ".cu":
            continue
        alias = path.stem
        if _IDENTIFIER_RE.fullmatch(alias) is None:
            raise CUDAExtensionError(
                f"CUDA source stem must be a C/Python identifier: {relative!r}"
            )
        previous = aliases.get(alias)
        if previous is not None and previous != relative:
            raise CUDAExtensionError(
                f"CUDA compilation units {previous!r} and {relative!r} share import alias {alias!r}"
            )
        aliases[alias] = relative
        units.append((path, relative, alias))
    return units, inventory, selected


def _compiler_environment() -> dict[str, str]:
    path = os.environ.get("OPTIMA_BUILD_PATH", os.environ.get("PATH", "")).strip()
    if not path:
        raise CUDAExtensionError("validator-owned compiler PATH is empty")
    components = path.split(os.pathsep)
    if any(not part or not Path(part).is_absolute() for part in components):
        raise CUDAExtensionError("validator-owned compiler PATH must contain absolute entries")
    temporary = os.environ.get("OPTIMA_BUILD_TMPDIR", "/tmp").strip()
    if not temporary or not Path(temporary).is_absolute():
        raise CUDAExtensionError("OPTIMA_BUILD_TMPDIR must be an absolute container path")
    # Construct, do not copy: CPATH, NVCC_*, LD_*, compiler overrides, Python
    # paths, proxies, credentials, and user configuration cannot reach nvcc.
    return {
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": path,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "SOURCE_DATE_EPOCH": "0",
        "TEMP": temporary,
        "TMP": temporary,
        "TMPDIR": temporary,
        "TZ": "UTC",
    }


def _tool_identity(name: str, env: dict[str, str]) -> dict[str, str]:
    found = shutil.which(name, path=env["PATH"])
    if found is None:
        raise CUDAExtensionError(f"required CUDA toolchain executable is missing: {name}")
    try:
        path = Path(found).resolve(strict=True)
    except OSError as exc:
        raise CUDAExtensionError(f"cannot resolve toolchain executable {name}: {exc}") from None
    if not path.is_file():
        raise CUDAExtensionError(f"toolchain executable is not a regular file: {path}")
    try:
        completed = subprocess.run(
            [str(path), "--version"],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            timeout=30,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise CUDAExtensionError(f"cannot identify toolchain executable {path}: {exc}") from None
    return {
        "path": str(path),
        "sha256": _sha256_file(path, single_link=False),
        "version": completed.stdout.strip(),
    }


def _pinned_roots(context_roots: tuple[Path, ...], *, production: bool) -> list[str]:
    raw = os.environ.get("OPTIMA_PINNED_BUILD_ROOTS", "").strip()
    if production and not raw:
        raise CUDAExtensionError("build phase requires OPTIMA_PINNED_BUILD_ROOTS")
    roots: list[Path] = []
    for value in raw.split(os.pathsep) if raw else ():
        candidate = Path(value)
        if not candidate.is_absolute():
            raise CUDAExtensionError("pinned build roots must be absolute container paths")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise CUDAExtensionError(f"pinned build root is unavailable: {candidate}: {exc}") from None
        if not resolved.is_dir() or resolved == Path("/"):
            raise CUDAExtensionError(f"pinned build root is unsafe: {candidate}")
        roots.append(resolved)
    if not production:
        roots.extend(root for root in context_roots if root.is_dir())
        for default in (Path("/usr/include"), Path("/usr/lib")):
            if default.is_dir():
                roots.append(default.resolve())
    canonical = sorted({str(root) for root in roots})
    if not canonical:
        raise CUDAExtensionError("native build has no image-bound dependency roots")
    return canonical


def _build_context(architecture: str, env: dict[str, str], *, production: bool) -> dict[str, object]:
    # Importing Torch is allowed here: build runs inside the disposable image, and
    # load runs inside an untrusted engine rank.  This module is never imported by
    # the trusted host controller.
    import torch

    torch_root = Path(torch.__file__).resolve().parent
    python_include = Path(sysconfig.get_paths()["include"]).resolve()
    torch_include = (torch_root / "include").resolve()
    torch_api_include = (torch_include / "torch" / "csrc" / "api" / "include").resolve()
    torch_lib = (torch_root / "lib").resolve()
    nvcc = _tool_identity("nvcc", env)
    ptxas = _tool_identity("ptxas", env)
    cuda_include = Path(nvcc["path"]).parent.parent / "include"
    roots = _pinned_roots(
        (python_include, torch_include, torch_api_include, torch_lib, cuda_include.resolve()),
        production=production,
    )
    return {
        "compiler_env_digest": _canonical_hash(env),
        "cxx11_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
        "link_libraries": list(_LINK_LIBRARIES),
        "nvcc": nvcc,
        "nvcc_architecture": _nvcc_architecture(architecture),
        "nvcc_flags": list(_COMPILE_FLAGS),
        "pinned_build_roots": roots,
        "ptxas": ptxas,
        "python_include": str(python_include),
        "python_soabi": str(sysconfig.get_config_var("SOABI") or ""),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "torch_api_include": str(torch_api_include),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "torch_include": str(torch_include),
        "torch_lib": str(torch_lib),
        "torch_version": str(torch.__version__),
    }


def _validate_context(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != _CONTEXT_FIELDS:
        raise CUDAExtensionError("native extension build_context schema mismatch")
    for tool_name in ("nvcc", "ptxas"):
        tool = value[tool_name]
        if not isinstance(tool, dict) or set(tool) != _TOOL_FIELDS:
            raise CUDAExtensionError(f"native extension {tool_name} receipt schema mismatch")
        _require_digest(tool["sha256"], field=f"{tool_name} sha256")
        if not all(isinstance(tool[key], str) for key in _TOOL_FIELDS):
            raise CUDAExtensionError(f"native extension {tool_name} receipt is malformed")
    if value["nvcc_flags"] != list(_COMPILE_FLAGS):
        raise CUDAExtensionError("native extension compiler flags differ from policy")
    if value["link_libraries"] != list(_LINK_LIBRARIES):
        raise CUDAExtensionError("native extension link libraries differ from policy")
    roots = value["pinned_build_roots"]
    if not isinstance(roots, list) or not roots or not all(
        isinstance(root, str) and Path(root).is_absolute() and root != "/" for root in roots
    ):
        raise CUDAExtensionError("native extension pinned build roots are malformed")
    if roots != sorted(set(roots)):
        raise CUDAExtensionError("native extension pinned build roots are not canonical")
    _require_digest(value["compiler_env_digest"], field="compiler_env_digest")
    return value


def _artifact_identity(
    *,
    build_spec_digest: str,
    tree_digest: str,
    architecture: str,
    source: str,
    alias: str,
    source_inventory: list[dict[str, str]],
    patcher_sha256: str,
    context: dict[str, object],
) -> tuple[str, str]:
    payload = {
        "build_context": context,
        "build_spec_digest": build_spec_digest,
        "patcher_sha256": patcher_sha256,
        "schema": _SCHEMA,
        "source": source,
        "source_inventory": source_inventory,
        "target_architecture": architecture,
        "tree_digest": tree_digest,
    }
    artifact_id = _canonical_hash(payload)
    # A fixed-length validator name avoids miner-controlled path length and still
    # gives every PyInit symbol a collision-resistant identity.  ``alias`` is added
    # to sys.modules only after the exact native module has loaded.
    module_name = f"optima_cuda_{artifact_id}"
    return artifact_id, module_name


def _depfile_tokens(raw: str) -> list[str]:
    logical = raw.replace("\\\r\n", "").replace("\\\n", "")
    colon = -1
    escaped = False
    for index, character in enumerate(logical):
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == ":":
            colon = index
            break
    if colon < 1:
        raise CUDAExtensionError("compiler dependency file lacks a target separator")
    result: list[str] = []
    token: list[str] = []
    escaped = False
    for character in logical[colon + 1 :]:
        if escaped:
            token.append(character)
            escaped = False
        elif character == "\\":
            escaped = True
        elif character.isspace():
            if token:
                result.append("".join(token))
                token.clear()
        else:
            token.append(character)
    if escaped:
        raise CUDAExtensionError("compiler dependency file ends in an escape")
    if token:
        result.append("".join(token))
    if not result:
        raise CUDAExtensionError("compiler dependency file names no inputs")
    return result


def _relative_to(path: Path, root: Path) -> str | None:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return None


def _dependencies(
    depfile: Path,
    *,
    bundle: Path,
    source: str,
    selected_sources: frozenset[str],
    pinned_roots: list[str],
) -> list[dict[str, str]]:
    try:
        info = depfile.lstat()
        if (
            stat.S_ISLNK(info.st_mode)
            or not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or info.st_size < 1
            or info.st_size > _MAX_DEPFILE_BYTES
        ):
            raise CUDAExtensionError("compiler dependency file has an unsafe shape")
        raw = depfile.read_text(encoding="utf-8")
    except CUDAExtensionError:
        raise
    except (OSError, UnicodeError) as exc:
        raise CUDAExtensionError(f"cannot read compiler dependency file: {exc}") from None
    roots = [Path(root).resolve(strict=True) for root in pinned_roots]
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for token in _depfile_tokens(raw):
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = bundle / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise CUDAExtensionError(f"compiler dependency is unavailable: {token!r}: {exc}") from None
        relative = _relative_to(resolved, bundle)
        if relative is not None:
            if relative not in selected_sources:
                raise CUDAExtensionError(
                    "compiler dependency is an undeclared or off-domain native "
                    f"source for this architecture: {relative}"
                )
            row = {
                "kind": "engine_tree",
                "path": relative,
                "root": ".",
                "sha256": _sha256_file(resolved),
            }
        else:
            match: tuple[int, Path, str] | None = None
            for index, root in enumerate(roots):
                under = _relative_to(resolved, root)
                if under is not None:
                    match = (index, root, under)
                    break
            if match is None:
                raise CUDAExtensionError(
                    f"compiler dependency is outside the engine tree and pinned image roots: {resolved}"
                )
            index, root, under = match
            row = {
                "kind": "image",
                "path": under,
                "root": f"{index}:{root}",
                "sha256": _sha256_file(resolved, single_link=False),
            }
        rows[(row["kind"], row["root"], row["path"])] = row
    result = [rows[key] for key in sorted(rows)]
    if not any(row["kind"] == "engine_tree" and row["path"] == source for row in result):
        raise CUDAExtensionError("compiler dependency file omits the declared compilation unit")
    return result


def _compile(
    *,
    bundle: Path,
    source: str,
    output: Path,
    depfile: Path,
    module_name: str,
    context: dict[str, object],
    env: dict[str, str],
) -> None:
    nvcc = context["nvcc"]
    assert isinstance(nvcc, dict)
    command = [
        str(nvcc["path"]),
        *_COMPILE_FLAGS,
        f"-arch={context['nvcc_architecture']}",
        "-MD",
        "-MF",
        str(depfile),
        "-MT",
        module_name,
        f"-DTORCH_EXTENSION_NAME={module_name}",
        f"-D_GLIBCXX_USE_CXX11_ABI={context['cxx11_abi']}",
        f"-I{context['torch_include']}",
        f"-I{context['torch_api_include']}",
        f"-I{context['python_include']}",
        source,
        f"-L{context['torch_lib']}",
        *[f"-l{name}" for name in _LINK_LIBRARIES],
        "-o",
        str(output),
    ]
    _log("compiling declared CUDA unit " + source)
    timeout = float(os.environ.get("OPTIMA_NATIVE_COMPILE_TIMEOUT_S", "1200"))
    if not (0 < timeout <= 7200):
        raise CUDAExtensionError("OPTIMA_NATIVE_COMPILE_TIMEOUT_S is outside policy")
    try:
        subprocess.run(
            command,
            check=True,
            cwd=bundle,
            env=env,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
            close_fds=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise CUDAExtensionError(f"CUDA compilation timed out for {source}") from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CUDAExtensionError(f"CUDA compilation failed for {source}: {exc}") from exc


def _write_json_exclusive(path: Path, value: object) -> None:
    raw = _canonical_bytes(value) + b"\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_set(
    *,
    bundle: Path,
    output_root: Path,
    build_spec_digest: str,
    tree_digest: str,
    architecture: str,
    units: list[tuple[Path, str, str]],
    source_inventory: list[dict[str, str]],
    selected_sources: frozenset[str],
    context: dict[str, object],
    env: dict[str, str],
) -> dict[str, object]:
    cuda_root = output_root / _ARTIFACT_DIR
    try:
        cuda_root.mkdir(mode=0o700)
    except FileExistsError:
        raise CUDAExtensionError("native artifact stage already contains a cuda product") from None
    patcher_sha256 = _patcher_hash()
    rows: list[dict[str, object]] = []
    for _source_path, source, alias in units:
        artifact_id, module_name = _artifact_identity(
            build_spec_digest=build_spec_digest,
            tree_digest=tree_digest,
            architecture=architecture,
            source=source,
            alias=alias,
            source_inventory=source_inventory,
            patcher_sha256=patcher_sha256,
            context=context,
        )
        unit_root = cuda_root / artifact_id
        unit_root.mkdir(mode=0o700)
        artifact_name = f"{module_name}.so"
        depfile_name = "dependencies.d"
        temporary_so = unit_root / f".{artifact_name}.{uuid.uuid4().hex}.tmp"
        temporary_dep = unit_root / f".{depfile_name}.{uuid.uuid4().hex}.tmp"
        artifact = unit_root / artifact_name
        depfile = unit_root / depfile_name
        try:
            _compile(
                bundle=bundle,
                source=source,
                output=temporary_so,
                depfile=temporary_dep,
                module_name=module_name,
                context=context,
                env=env,
            )
            if not temporary_so.is_file() or temporary_so.is_symlink():
                raise CUDAExtensionError(f"compiler produced no regular artifact for {source}")
            dependencies = _dependencies(
                temporary_dep,
                bundle=bundle,
                source=source,
                selected_sources=selected_sources,
                pinned_roots=list(context["pinned_build_roots"]),  # type: ignore[arg-type]
            )
            os.replace(temporary_so, artifact)
            os.replace(temporary_dep, depfile)
        finally:
            temporary_so.unlink(missing_ok=True)
            temporary_dep.unlink(missing_ok=True)
        rows.append(
            {
                "alias": alias,
                "artifact_id": artifact_id,
                "artifact_path": f"{_ARTIFACT_DIR}/{artifact_id}/{artifact_name}",
                "artifact_sha256": _sha256_file(artifact),
                "dependencies": dependencies,
                "depfile_path": f"{_ARTIFACT_DIR}/{artifact_id}/{depfile_name}",
                "depfile_sha256": _sha256_file(depfile),
                "module_name": module_name,
                "source": source,
            }
        )
    index: dict[str, object] = {
        "build_context": context,
        "build_spec_digest": build_spec_digest,
        "patcher_sha256": patcher_sha256,
        "schema": _SCHEMA,
        "selected_sources": sorted(selected_sources),
        "source_inventory": source_inventory,
        "target_architecture": architecture,
        "tree_digest": tree_digest,
        "units": rows,
    }
    _write_json_exclusive(cuda_root / _INDEX_NAME, index)
    return index


def _validate_source_inventory(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise CUDAExtensionError("native extension source inventory must be an array")
    result: list[dict[str, str]] = []
    for row in value:
        if not isinstance(row, dict) or set(row) != _SOURCE_FIELDS:
            raise CUDAExtensionError("native extension source inventory row is malformed")
        path = row["path"]
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in Path(path).parts:
            raise CUDAExtensionError("native extension source inventory path is unsafe")
        result.append({"path": path, "sha256": _require_digest(row["sha256"], field="source sha256")})
    if result != sorted(result, key=lambda row: row["path"]):
        raise CUDAExtensionError("native extension source inventory is not canonical")
    return result


def _validate_index(
    *,
    root: Path,
    bundle: Path,
    expected_build_spec: str,
    expected_tree: str,
    expected_architecture: str,
    current_units: list[tuple[Path, str, str]],
    current_inventory: list[dict[str, str]],
    current_selected_sources: frozenset[str],
) -> list[tuple[str, str, Path]]:
    value = _read_json(root / _ARTIFACT_DIR / _INDEX_NAME)
    if not isinstance(value, dict) or set(value) != _INDEX_FIELDS:
        raise CUDAExtensionError("native extension index schema mismatch")
    if value["schema"] != _SCHEMA:
        raise CUDAExtensionError("native extension index version mismatch")
    if value["build_spec_digest"] != expected_build_spec:
        raise CUDAExtensionError("native extension build-spec identity mismatch")
    if value["tree_digest"] != expected_tree:
        raise CUDAExtensionError("native extension tree identity mismatch")
    if value["target_architecture"] != expected_architecture:
        raise CUDAExtensionError("native extension target architecture mismatch")
    patcher_sha256 = _require_digest(value["patcher_sha256"], field="patcher_sha256")
    if patcher_sha256 != _patcher_hash():
        raise CUDAExtensionError("native extension patcher identity mismatch")
    context = _validate_context(value["build_context"])
    inventory = _validate_source_inventory(value["source_inventory"])
    if inventory != current_inventory:
        raise CUDAExtensionError(
            "native extension source inventory differs from engine tree"
        )
    selected_sources = value["selected_sources"]
    inventory_paths = {row["path"] for row in current_inventory}
    if not isinstance(selected_sources, list) or any(
        not isinstance(path, str) for path in selected_sources
    ):
        raise CUDAExtensionError(
            "native extension architecture-selected source inventory mismatch"
        )
    if (
        selected_sources != sorted(set(selected_sources))
        or any(path not in inventory_paths for path in selected_sources)
        or selected_sources != sorted(current_selected_sources)
    ):
        raise CUDAExtensionError(
            "native extension architecture-selected source inventory mismatch"
        )
    rows = value["units"]
    if not isinstance(rows, list) or len(rows) != len(current_units):
        raise CUDAExtensionError("native extension unit inventory mismatch")
    expected_by_source = {relative: alias for _, relative, alias in current_units}
    observed_sources: list[str] = []
    loadable: list[tuple[str, str, Path]] = []
    allowed_paths = {f"{_ARTIFACT_DIR}/{_INDEX_NAME}"}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _UNIT_FIELDS:
            raise CUDAExtensionError("native extension unit row schema mismatch")
        source = row["source"]
        alias = row["alias"]
        if not isinstance(source, str) or not isinstance(alias, str):
            raise CUDAExtensionError("native extension source/alias is malformed")
        if expected_by_source.get(source) != alias:
            raise CUDAExtensionError("native extension source/alias differs from engine tree")
        artifact_id, module_name = _artifact_identity(
            build_spec_digest=expected_build_spec,
            tree_digest=expected_tree,
            architecture=expected_architecture,
            source=source,
            alias=alias,
            source_inventory=current_inventory,
            patcher_sha256=patcher_sha256,
            context=context,
        )
        if row["artifact_id"] != artifact_id or row["module_name"] != module_name:
            raise CUDAExtensionError("native extension unit identity mismatch")
        artifact_rel = f"{_ARTIFACT_DIR}/{artifact_id}/{module_name}.so"
        depfile_rel = f"{_ARTIFACT_DIR}/{artifact_id}/dependencies.d"
        if row["artifact_path"] != artifact_rel or row["depfile_path"] != depfile_rel:
            raise CUDAExtensionError("native extension unit paths are noncanonical")
        artifact = root / artifact_rel
        depfile = root / depfile_rel
        if _sha256_file(artifact) != _require_digest(row["artifact_sha256"], field="artifact sha256"):
            raise CUDAExtensionError("native extension artifact hash mismatch")
        if _sha256_file(depfile) != _require_digest(row["depfile_sha256"], field="depfile sha256"):
            raise CUDAExtensionError("native extension depfile hash mismatch")
        dependencies = row["dependencies"]
        if not isinstance(dependencies, list) or not all(
            isinstance(item, dict) and set(item) == _DEPENDENCY_FIELDS for item in dependencies
        ):
            raise CUDAExtensionError("native extension dependency receipt is malformed")
        actual_dependencies = _dependencies(
            depfile,
            bundle=bundle,
            source=source,
            selected_sources=current_selected_sources,
            pinned_roots=list(context["pinned_build_roots"]),  # type: ignore[arg-type]
        )
        if dependencies != actual_dependencies:
            raise CUDAExtensionError("native extension dependency receipt mismatch")
        observed_sources.append(source)
        allowed_paths.update({artifact_rel, depfile_rel})
        loadable.append((alias, module_name, artifact))
    if observed_sources != sorted(expected_by_source):
        raise CUDAExtensionError("native extension units are not canonical")
    actual_cuda_paths = {
        path.relative_to(root).as_posix()
        for path in (root / _ARTIFACT_DIR).rglob("*")
        if path.is_file()
    }
    if actual_cuda_paths != allowed_paths:
        raise CUDAExtensionError("native extension publication contains unexpected cuda files")
    return loadable


def _load(alias: str, module_name: str, artifact: Path) -> None:
    import torch  # noqa: F401 - the extension links against libtorch

    existing_alias = sys.modules.get(alias)
    existing_native = sys.modules.get(module_name)
    if existing_alias is not None and existing_alias is not existing_native:
        raise CUDAExtensionError(f"refusing CUDA import alias collision: {alias!r}")
    if existing_native is None:
        specification = importlib.util.spec_from_file_location(module_name, artifact)
        if specification is None or specification.loader is None:
            raise CUDAExtensionError(f"cannot create native loader for {artifact}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        try:
            specification.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(module_name, None)
            raise
    else:
        loaded = Path(str(getattr(existing_native, "__file__", ""))).resolve()
        if loaded != artifact.resolve():
            raise CUDAExtensionError(
                f"native module collision for {module_name!r}: loaded {loaded}, expected {artifact}"
            )
        module = existing_native
    sys.modules[alias] = module
    _log(f"loaded {module_name} as {alias}")


def _build_phase(
    bundle: Path,
    build_spec: str,
    tree_digest: str,
    architecture: str,
    units: list[tuple[Path, str, str]],
    inventory: list[dict[str, str]],
    selected_sources: frozenset[str],
) -> None:
    if os.environ.get("OPTIMA_REBUILD_CONTAINER") != "1":
        raise CUDAExtensionError("native build phase requires the disposable rebuild container")
    for name in (
        "OPTIMA_BUILD_PATH",
        "OPTIMA_BUILD_TMPDIR",
        "OPTIMA_NATIVE_COMPILE_TIMEOUT_S",
    ):
        if not os.environ.get(name, "").strip():
            raise CUDAExtensionError(f"native build phase requires {name}")
    raw_stage = os.environ.get("OPTIMA_NATIVE_ARTIFACT_STAGE", "").strip()
    if not raw_stage or not Path(raw_stage).is_absolute():
        raise CUDAExtensionError("build phase requires absolute OPTIMA_NATIVE_ARTIFACT_STAGE")
    stage = Path(raw_stage)
    if stage.is_symlink() or not stage.is_dir():
        raise CUDAExtensionError("native artifact stage must be an existing real directory")
    env = _compiler_environment()
    context = _build_context(architecture, env, production=True)
    _build_set(
        bundle=bundle,
        output_root=stage,
        build_spec_digest=build_spec,
        tree_digest=tree_digest,
        architecture=architecture,
        units=units,
        source_inventory=inventory,
        selected_sources=selected_sources,
        context=context,
        env=env,
    )
    _log(f"built {len(units)} CUDA extension unit(s); native loading deferred")


def _load_phase(
    bundle: Path,
    build_spec: str,
    tree_digest: str,
    architecture: str,
    units: list[tuple[Path, str, str]],
    inventory: list[dict[str, str]],
    selected_sources: frozenset[str],
) -> None:
    if os.environ.get("OPTIMA_ENGINE_WORKER") != "1":
        raise CUDAExtensionError("native load phase requires an isolated engine worker")
    raw_root = os.environ.get("OPTIMA_NATIVE_ARTIFACT_ROOT", "").strip()
    publication_digest = _require_digest(
        os.environ.get("OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST", "").strip(),
        field="OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST",
    )
    if not raw_root or not Path(raw_root).is_absolute():
        raise CUDAExtensionError("load phase requires absolute OPTIMA_NATIVE_ARTIFACT_ROOT")
    from optima.eval.native_artifact import reopen_native_artifact

    publication = reopen_native_artifact(
        raw_root,
        expected_build_spec_digest=build_spec,
        expected_publication_digest=publication_digest,
        expected_owner_uid=None,
    )
    loadable = _validate_index(
        root=publication.root,
        bundle=bundle,
        expected_build_spec=build_spec,
        expected_tree=tree_digest,
        expected_architecture=architecture,
        current_units=units,
        current_inventory=inventory,
        current_selected_sources=selected_sources,
    )
    # Validate the complete set before executing any native initializer.
    for alias, module_name, artifact in loadable:
        _load(alias, module_name, artifact)


def _development_all(
    bundle: Path,
    build_spec: str,
    tree_digest: str,
    architecture: str,
) -> None:
    _log("combined build+load is a non-authoritative development path")
    env = _compiler_environment()
    if any(shutil.which(tool, path=env["PATH"]) is None for tool in ("nvcc", "ptxas")):
        # Coordinator/nsenter launches intentionally inherit a minimal host PATH.
        # Resolve the container's validator-owned CUDA root without changing the
        # process environment; the exact executables and the resulting compiler
        # environment are still hashed into the development artifact identity.
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda").strip()
        cuda_bin: Path | None = None
        if cuda_home and Path(cuda_home).is_absolute():
            candidate = Path(cuda_home) / "bin"
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                resolved = None
            if resolved is not None and resolved.is_dir():
                cuda_bin = resolved
        if cuda_bin is not None:
            env = dict(env)
            env["PATH"] = os.pathsep.join((str(cuda_bin), env["PATH"]))
    missing_tools = tuple(
        tool
        for tool in ("nvcc", "ptxas")
        if shutil.which(tool, path=env["PATH"]) is None
    )
    if missing_tools:
        if architecture:
            raise CUDAExtensionError(
                "sealed development target lacks required CUDA toolchain "
                f"{missing_tools!r}"
            )
        _log("CUDA toolchain unavailable; skipping native development product")
        return
    if not architecture:
        # Direct development may discover an architecture from a live device,
        # but a validator-projected target is already sufficient to compile.
        # Spawned engine interpreters can import the bootstrap before SGLang
        # establishes their CUDA context; requiring torch.cuda here therefore
        # incorrectly turns a sealed target into a silent stock fallback.
        import torch

        if not torch.cuda.is_available():
            _log("CUDA device unavailable; skipping native development product")
            return
        major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
        architecture = _canonical_architecture(f"sm{major}{minor}")
    units, inventory, selected_sources = _declared_sources(bundle, architecture)
    context = _build_context(architecture, env, production=False)
    if not tree_digest:
        from optima.bundle_hash import content_hash

        tree_digest = content_hash(bundle)
    if not build_spec:
        build_spec = _canonical_hash(
            {
                "build_context": context,
                "patcher_sha256": _patcher_hash(),
                "schema": "optima.cuda-extension-development.v1",
                "target_architecture": architecture,
                "tree_digest": tree_digest,
            }
        )
    raw_cache = os.environ.get("OPTIMA_CUDA_EXT_CACHE", "").strip()
    cache_root = Path(raw_cache) if raw_cache else Path.home() / ".cache" / "optima" / "cuda_ext"
    destination = cache_root / "v3" / build_spec[:2] / build_spec
    lock = cache_root / "v3" / ".locks" / f"{build_spec}.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    import fcntl

    with lock.open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            loadable = _validate_index(
                root=destination,
                bundle=bundle,
                expected_build_spec=build_spec,
                expected_tree=tree_digest,
                expected_architecture=architecture,
                current_units=units,
                current_inventory=inventory,
                current_selected_sources=selected_sources,
            )
        except CUDAExtensionError:
            stage = destination.with_name(f".{build_spec}.{os.getpid()}.{uuid.uuid4().hex}.stage")
            stage.mkdir(parents=True)
            try:
                _build_set(
                    bundle=bundle,
                    output_root=stage,
                    build_spec_digest=build_spec,
                    tree_digest=tree_digest,
                    architecture=architecture,
                    units=units,
                    source_inventory=inventory,
                    selected_sources=selected_sources,
                    context=context,
                    env=env,
                )
                if destination.exists():
                    shutil.rmtree(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(stage, destination)
            finally:
                shutil.rmtree(stage, ignore_errors=True)
            loadable = _validate_index(
                root=destination,
                bundle=bundle,
                expected_build_spec=build_spec,
                expected_tree=tree_digest,
                expected_architecture=architecture,
                current_units=units,
                current_inventory=inventory,
                current_selected_sources=selected_sources,
            )
    for alias, module_name, artifact in loadable:
        _load(alias, module_name, artifact)


def main() -> None:
    bundle = _bundle_root()
    if bundle is None:
        return
    phase = _phase()
    build_spec, tree_digest, architecture = _production_identity(phase)
    if phase == "build":
        units, inventory, selected_sources = _declared_sources(bundle, architecture)
        _build_phase(
            bundle,
            build_spec,
            tree_digest,
            architecture,
            units,
            inventory,
            selected_sources,
        )
    elif phase == "load":
        units, inventory, selected_sources = _declared_sources(bundle, architecture)
        _load_phase(
            bundle,
            build_spec,
            tree_digest,
            architecture,
            units,
            inventory,
            selected_sources,
        )
    else:
        _development_all(bundle, build_spec, tree_digest, architecture)


if __name__ == "__main__":
    main()
