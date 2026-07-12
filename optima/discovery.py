"""Bounded, non-crownable discovery proposals for pinned SGLang source.

The discovery lane accepts only exact text patches against a validator-owned
SGLang policy.  A proposal is deliberately not an op bundle, catalog target,
stack contribution, release, or settlement record.  This module is data and
filesystem plumbing only: it parses the separate discovery ABI, freezes exact
proposal bytes, stages allowed diffs beneath the fixed native-build envelope,
and projects the result from a completely reopened immutable publication.

Cache selection, subprocesses, OCI mounts, process-role activation, evaluation,
and promotion authority live in their respective trusted controller layers.
"""

from __future__ import annotations

import ast
import json
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optima.engine_tree import MaterializedEngineTree

from optima.deppatch import DepPatchError, FilePatch, apply_file_patch, parse_patch_text
from optima.eval.native_artifact import NativeArtifactPublication
from optima.stack_identity import (
    StackIdentityError,
    canonical_digest,
    canonical_json_bytes,
    require_sha256_hex,
    sha256_hex,
)
from optima.stack_manifest import EvaluationStackManifest
from optima.stack_plan import StackArmIdentity


DISCOVERY_ABI_VERSION = "optima-discovery-abi-v1"
DISCOVERY_OVERLAY_SCHEMA = "optima.discovery-overlay.v1"
DISCOVERY_POLICY_ID = "sglang-inference-discovery-v1"
DISCOVERY_PROMOTIONS = frozenset(
    {"new_singleton", "atomic_target", "reviewed_engine_change", "bounty_only"}
)
# Keep the discovery authority on a static import closure. A focused drift test
# requires this value to match compat.PINNED_SGLANG after every deliberate bump.
_DEFAULT_SGLANG_VERSION = "0.5.13.post1"

_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SELECTOR_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/+\-]{0,255}\Z")
_ARCH_RE = re.compile(r"sm[0-9]{2,3}[a-z]?\Z")
_PATCH_SUFFIXES = frozenset({".patch", ".diff"})
_MANIFEST_FIELDS = frozenset(
    {
        "abi_version",
        "applicability",
        "build_profile",
        "bundle_id",
        "conflicts",
        "dependencies",
        "patches",
        "requested_promotion",
    }
)
_APPLICABILITY_FIELDS = frozenset(
    {"architectures", "arenas", "models", "tensor_parallel_sizes"}
)
_IGNORED_DIRECTORIES = frozenset({".claude", ".git", "__pycache__"})
_IGNORED_FILES = frozenset({".clang-format", ".claude"})
_IGNORED_SUFFIXES = frozenset({".pyc", ".pyo"})
_STAMP_FIELDS = frozenset({"identity", "schema"})
_DISCOVERY_ENVELOPE = "dep_overlays/discovery"
_DISCOVERY_STAMP = f"{_DISCOVERY_ENVELOPE}/overlay.json"
_DISCOVERY_PACKAGE = f"{_DISCOVERY_ENVELOPE}/site/sglang/"
_DISCOVERY_ENGINE_METADATA = "metadata/optima_discovery.json"
_DISCOVERY_ENGINE_SCHEMA = "optima.discovery-engine-tree.v1"
_DISCOVERY_ENGINE_FIELDS = frozenset(
    {
        "build_profile", "build_profile_digest", "incumbent_stack_digest",
        "incumbent_tree_digest", "policy", "policy_digest", "proposal_digest",
        "proposal_files", "schema",
    }
)


class DiscoveryError(ValueError):
    """A discovery proposal, policy, plan, or overlay is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiscoveryError(message)


def _strict_object(value: object, fields: frozenset[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise DiscoveryError(f"{name} schema mismatch")
    return value


def _digest(value: object, *, field: str) -> str:
    try:
        digest = require_sha256_hex(value, field=field)
    except StackIdentityError as exc:
        raise DiscoveryError(str(exc)) from None
    if digest == "0" * 64:
        raise DiscoveryError(f"{field} must not be the all-zero digest")
    return digest


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise DiscoveryError(f"{field} must match [a-z0-9][a-z0-9._-]*")
    return value


def _selector(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SELECTOR_RE.fullmatch(value) is None:
        raise DiscoveryError(f"{field} is not a canonical selector")
    return value


def _sorted_unique_strings(
    value: object,
    *,
    field: str,
    required: bool = False,
    selector: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or isinstance(value, (str, bytes)):
        raise DiscoveryError(f"{field} must be an array")
    parser = _selector if selector else _identifier
    rows = tuple(parser(item, field=f"{field}[]") for item in value)
    if required and not rows:
        raise DiscoveryError(f"{field} must not be empty")
    if rows != tuple(sorted(set(rows))):
        raise DiscoveryError(f"{field} must be sorted and unique")
    return rows


def _canonical_relative(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise DiscoveryError(f"{field} must be a canonical relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise DiscoveryError(f"{field} must be a canonical relative path")
    try:
        value.encode("ascii", "strict")
    except UnicodeError:
        raise DiscoveryError(f"{field} must be ASCII") from None
    return value


def _load_toml_bytes(raw: bytes) -> Mapping[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DiscoveryError(f"manifest.toml is not UTF-8: {exc}") from None
    try:
        try:
            import tomllib  # type: ignore

            value = tomllib.loads(text)
        except ModuleNotFoundError:
            import tomli  # type: ignore

            value = tomli.loads(text)
    except Exception as exc:  # TOML backends do not share one error type
        raise DiscoveryError(f"failed to parse manifest.toml: {exc}") from None
    if not isinstance(value, Mapping):
        raise DiscoveryError("manifest.toml must contain a top-level table")
    return value


@dataclass(frozen=True)
class DiscoveryApplicability:
    arenas: tuple[str, ...]
    models: tuple[str, ...]
    architectures: tuple[str, ...]
    tensor_parallel_sizes: tuple[int, ...]

    def __post_init__(self) -> None:
        arenas = _sorted_unique_strings(
            self.arenas, field="applicability.arenas", required=True
        )
        models = _sorted_unique_strings(
            self.models, field="applicability.models", required=True, selector=True
        )
        architectures = _sorted_unique_strings(
            self.architectures,
            field="applicability.architectures",
            required=True,
        )
        if any(_ARCH_RE.fullmatch(value) is None for value in architectures):
            raise DiscoveryError("applicability.architectures contains a non-SM architecture")
        sizes = self.tensor_parallel_sizes
        if not isinstance(sizes, (list, tuple)) or not sizes:
            raise DiscoveryError("applicability.tensor_parallel_sizes must not be empty")
        sizes = tuple(sizes)
        if any(type(value) is not int or value < 1 or value > 65_536 for value in sizes):
            raise DiscoveryError("applicability.tensor_parallel_sizes is invalid")
        if sizes != tuple(sorted(set(sizes))):
            raise DiscoveryError(
                "applicability.tensor_parallel_sizes must be sorted and unique"
            )
        object.__setattr__(self, "arenas", arenas)
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "architectures", architectures)
        object.__setattr__(self, "tensor_parallel_sizes", sizes)

    def to_dict(self) -> dict[str, object]:
        return {
            "architectures": list(self.architectures),
            "arenas": list(self.arenas),
            "models": list(self.models),
            "tensor_parallel_sizes": list(self.tensor_parallel_sizes),
        }


@dataclass(frozen=True)
class DiscoveryManifest:
    bundle_id: str
    abi_version: str
    build_profile: str
    patches: tuple[str, ...]
    applicability: DiscoveryApplicability
    dependencies: tuple[str, ...]
    conflicts: tuple[str, ...]
    requested_promotion: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "bundle_id", _identifier(self.bundle_id, field="bundle_id"))
        if self.abi_version != DISCOVERY_ABI_VERSION:
            raise DiscoveryError(
                f"unsupported abi_version {self.abi_version!r}; expected "
                f"{DISCOVERY_ABI_VERSION!r}"
            )
        object.__setattr__(
            self, "build_profile", _identifier(self.build_profile, field="build_profile")
        )
        if not isinstance(self.applicability, DiscoveryApplicability):
            raise DiscoveryError("applicability must be DiscoveryApplicability")
        if not isinstance(self.patches, (list, tuple)) or isinstance(
            self.patches, (str, bytes)
        ):
            raise DiscoveryError("patches must be an array")
        patches = tuple(_canonical_relative(value, field="patches[]") for value in self.patches)
        if not patches or patches != tuple(sorted(set(patches))):
            raise DiscoveryError("patches must be a nonempty sorted unique array")
        if any(PurePosixPath(value).suffix not in _PATCH_SUFFIXES for value in patches):
            raise DiscoveryError("patches must use .patch or .diff suffixes")
        object.__setattr__(self, "patches", patches)
        dependencies = _sorted_unique_strings(self.dependencies, field="dependencies")
        conflicts = _sorted_unique_strings(self.conflicts, field="conflicts")
        if set(dependencies) & set(conflicts):
            raise DiscoveryError("dependencies and conflicts must be disjoint")
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "conflicts", conflicts)
        if (
            not isinstance(self.requested_promotion, str)
            or self.requested_promotion not in DISCOVERY_PROMOTIONS
        ):
            raise DiscoveryError(
                f"requested_promotion must be one of {tuple(sorted(DISCOVERY_PROMOTIONS))!r}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "abi_version": self.abi_version,
            "applicability": self.applicability.to_dict(),
            "build_profile": self.build_profile,
            "bundle_id": self.bundle_id,
            "conflicts": list(self.conflicts),
            "dependencies": list(self.dependencies),
            "patches": list(self.patches),
            "requested_promotion": self.requested_promotion,
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.discovery.manifest", self.to_dict())


@dataclass(frozen=True)
class DiscoveryFile:
    path: str
    sha256: str
    size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _canonical_relative(self.path, field="file.path"))
        object.__setattr__(self, "sha256", _digest(self.sha256, field="file.sha256"))
        if type(self.size) is not int or self.size < 0:
            raise DiscoveryError("file.size must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size": self.size}

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryFile":
        row = _strict_object(
            value, frozenset({"path", "sha256", "size"}), name="discovery file"
        )
        return cls(path=row["path"], sha256=row["sha256"], size=row["size"])  # type: ignore[arg-type]


@dataclass(frozen=True)
class InspectedDiscovery:
    root: Path
    manifest: DiscoveryManifest
    files: tuple[DiscoveryFile, ...]
    patch_texts: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        root = Path(self.root)
        if not root.is_absolute():
            raise DiscoveryError("inspected discovery root must be absolute")
        object.__setattr__(self, "root", root)
        if not isinstance(self.manifest, DiscoveryManifest):
            raise DiscoveryError("inspected discovery manifest is invalid")
        files = tuple(self.files)
        if tuple(row.path for row in files) != tuple(sorted(row.path for row in files)):
            raise DiscoveryError("proposal inventory must be path-sorted")
        if len({row.path for row in files}) != len(files):
            raise DiscoveryError("proposal inventory contains duplicate paths")
        object.__setattr__(self, "files", files)
        patch_texts = tuple(self.patch_texts)
        if tuple(path for path, _text in patch_texts) != self.manifest.patches:
            raise DiscoveryError("frozen patch text order differs from manifest")
        by_path = {row.path: row for row in files}
        for path, text in patch_texts:
            if not isinstance(text, str):
                raise DiscoveryError("frozen patch contents must be text")
            raw = text.encode("utf-8")
            row = by_path.get(path)
            if row is None or row.size != len(raw) or row.sha256 != sha256_hex(raw):
                raise DiscoveryError(f"frozen patch contents differ from inventory: {path!r}")
        object.__setattr__(self, "patch_texts", patch_texts)

    def to_dict(self) -> dict[str, object]:
        return {
            "files": [row.to_dict() for row in self.files],
            "manifest": self.manifest.to_dict(),
        }

    @property
    def proposal_digest(self) -> str:
        return canonical_digest("optima.discovery.proposal", self.to_dict())


def _stable_read(path: Path, *, field: str, limit: int | None = None) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise DiscoveryError(f"cannot stat {field}: {exc}") from None
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise DiscoveryError(f"{field} must be a regular single-linked file")
    if limit is not None and before.st_size > limit:
        raise DiscoveryError(f"{field} exceeds {limit} bytes")
    try:
        first = path.read_bytes()
        middle = path.lstat()
        second = path.read_bytes()
        after = path.lstat()
    except OSError as exc:
        raise DiscoveryError(f"cannot read {field}: {exc}") from None
    signature = lambda row: (
        row.st_dev,
        row.st_ino,
        row.st_mode,
        row.st_nlink,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )
    if signature(before) != signature(middle) or signature(middle) != signature(after):
        raise DiscoveryError(f"{field} changed while being read")
    if first != second:
        raise DiscoveryError(f"{field} bytes changed while being read")
    return first


def _source_root(value: str | Path, *, field: str) -> Path:
    unresolved = Path(value)
    try:
        info = unresolved.lstat()
    except OSError as exc:
        raise DiscoveryError(f"cannot stat {field}: {exc}") from None
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise DiscoveryError(f"{field} must be a regular non-symlink directory")
    try:
        return unresolved.resolve(strict=True)
    except OSError as exc:
        raise DiscoveryError(f"cannot resolve {field}: {exc}") from None


def load_discovery_manifest(bundle_root: str | Path) -> DiscoveryManifest:
    """Parse only the closed discovery ABI; normal component intake stays separate."""

    root = _source_root(bundle_root, field="discovery root")
    raw = _stable_read(root / "manifest.toml", field="manifest.toml", limit=1 << 20)
    data = _strict_object(_load_toml_bytes(raw), _MANIFEST_FIELDS, name="discovery manifest")
    applicability_row = _strict_object(
        data["applicability"], _APPLICABILITY_FIELDS, name="discovery applicability"
    )
    applicability = DiscoveryApplicability(
        arenas=applicability_row["arenas"],  # type: ignore[arg-type]
        models=applicability_row["models"],  # type: ignore[arg-type]
        architectures=applicability_row["architectures"],  # type: ignore[arg-type]
        tensor_parallel_sizes=applicability_row["tensor_parallel_sizes"],  # type: ignore[arg-type]
    )
    manifest = DiscoveryManifest(
        bundle_id=data["bundle_id"],  # type: ignore[arg-type]
        abi_version=data["abi_version"],  # type: ignore[arg-type]
        build_profile=data["build_profile"],  # type: ignore[arg-type]
        patches=data["patches"],  # type: ignore[arg-type]
        applicability=applicability,
        dependencies=data["dependencies"],  # type: ignore[arg-type]
        conflicts=data["conflicts"],  # type: ignore[arg-type]
        requested_promotion=data["requested_promotion"],  # type: ignore[arg-type]
    )
    for relative in manifest.patches:
        path = root.joinpath(*PurePosixPath(relative).parts)
        _stable_read(path, field=f"patch {relative!r}", limit=16 << 20)
    return manifest


def _proposal_inventory(
    root: Path, *, declared_files: frozenset[str]
) -> tuple[DiscoveryFile, ...]:
    rows: list[DiscoveryFile] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted(directories):
            child = current_path / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise DiscoveryError(f"proposal contains unsafe directory entry: {child}")
            relative = _canonical_relative(
                child.relative_to(root).as_posix(), field="proposal directory path"
            )
            prefix = relative + "/"
            if not any(path.startswith(prefix) for path in declared_files):
                raise DiscoveryError(f"proposal contains undeclared directory: {relative!r}")
        for name in sorted(files):
            child = current_path / name
            relative = _canonical_relative(
                child.relative_to(root).as_posix(), field="proposal inventory path"
            )
            raw = _stable_read(child, field=f"proposal file {relative!r}")
            rows.append(DiscoveryFile(relative, sha256_hex(raw), len(raw)))
    rows.sort(key=lambda row: row.path)
    return tuple(rows)


def inspect_discovery(bundle_root: str | Path) -> InspectedDiscovery:
    """Freeze exact manifest/patch bytes and reject every undeclared tree entry."""

    root = _source_root(bundle_root, field="discovery root")
    first_manifest = load_discovery_manifest(root)
    expected = {"manifest.toml", *first_manifest.patches}
    first_files = _proposal_inventory(root, declared_files=frozenset(expected))
    actual = {row.path for row in first_files}
    if actual != expected:
        raise DiscoveryError(
            f"proposal file inventory differs from declaration; extra={tuple(sorted(actual - expected))!r}, "
            f"missing={tuple(sorted(expected - actual))!r}"
        )
    patch_texts: list[tuple[str, str]] = []
    for relative in first_manifest.patches:
        raw = _stable_read(
            root.joinpath(*PurePosixPath(relative).parts),
            field=f"patch {relative!r}",
            limit=16 << 20,
        )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DiscoveryError(f"patch {relative!r} is not UTF-8: {exc}") from None
        try:
            parse_patch_text(text)
        except DepPatchError as exc:
            raise DiscoveryError(f"patch {relative!r} rejected: {exc}") from None
        patch_texts.append((relative, text))
    second_manifest = load_discovery_manifest(root)
    second_files = _proposal_inventory(root, declared_files=frozenset(expected))
    if first_manifest != second_manifest or first_files != second_files:
        raise DiscoveryError("proposal changed while being inspected")
    return InspectedDiscovery(root, first_manifest, first_files, tuple(patch_texts))


@dataclass(frozen=True)
class DiscoveryPolicy:
    policy_id: str
    sglang_version: str
    allowed_prefixes: tuple[str, ...]
    allowed_files: tuple[str, ...]
    allowed_symbol_regions: tuple[str, ...]
    allowed_suffixes: tuple[str, ...]
    forbidden_path_markers: tuple[str, ...]
    forbidden_added_source: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _identifier(self.policy_id, field="policy_id"))
        _require(
            isinstance(self.sglang_version, str) and bool(self.sglang_version),
            "policy sglang_version must not be empty",
        )
        for field in (
            "allowed_prefixes",
            "allowed_files",
            "allowed_symbol_regions",
            "allowed_suffixes",
            "forbidden_path_markers",
            "forbidden_added_source",
        ):
            rows = tuple(getattr(self, field))
            if rows != tuple(sorted(set(rows))) or any(
                not isinstance(value, str) or not value for value in rows
            ):
                raise DiscoveryError(f"policy {field} must be sorted unique strings")
            object.__setattr__(self, field, rows)

    def to_dict(self) -> dict[str, object]:
        return {
            "allowed_files": list(self.allowed_files),
            "allowed_prefixes": list(self.allowed_prefixes),
            "allowed_suffixes": list(self.allowed_suffixes),
            "allowed_symbol_regions": list(self.allowed_symbol_regions),
            "forbidden_added_source": list(self.forbidden_added_source),
            "forbidden_path_markers": list(self.forbidden_path_markers),
            "policy_id": self.policy_id,
            "sglang_version": self.sglang_version,
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.discovery.policy", self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryPolicy":
        row = _strict_object(
            value, frozenset(cls.__dataclass_fields__), name="discovery policy"
        )
        try:
            result = cls(**row)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise DiscoveryError(f"discovery policy is invalid: {exc}") from None
        if result.to_dict() != dict(row):
            raise DiscoveryError("discovery policy serialization is not canonical")
        return result


DEFAULT_DISCOVERY_POLICY = DiscoveryPolicy(
    policy_id=DISCOVERY_POLICY_ID,
    sglang_version=_DEFAULT_SGLANG_VERSION,
    allowed_prefixes=tuple(sorted("""
        sglang/srt/batch_overlap/ sglang/srt/layers/ sglang/srt/mem_cache/
        sglang/srt/model_executor/ sglang/srt/models/
    """.split())),
    allowed_files=tuple(sorted("""
        sglang/srt/managers/overlap_utils.py
        sglang/srt/managers/prefill_delayer.py
        sglang/srt/managers/schedule_batch.py
        sglang/srt/managers/schedule_policy.py
        sglang/srt/managers/scheduler.py
        sglang/srt/managers/scheduler_components/dp_attn.py
        sglang/srt/managers/scheduler_components/flush_wrapper.py
        sglang/srt/managers/scheduler_components/idle_sleeper.py
        sglang/srt/managers/scheduler_components/invariant_checker.py
        sglang/srt/managers/scheduler_components/new_token_ratio_tracker.py
        sglang/srt/managers/scheduler_input_blocker.py
        sglang/srt/managers/scheduler_pp_mixin.py
        sglang/srt/managers/scheduler_recv_skipper.py
    """.split())),
    allowed_symbol_regions=tuple(
        sorted(
            [
                f"sglang/srt/managers/scheduler.py::Scheduler.{name}"
                for name in """
                    _build_hisparse_decode_batch _can_schedule_lora_req
                    _forward_isolation _get_new_batch_prefill_raw
                    _should_delay_dflash_prefill_for_batching event_loop_normal
                    event_loop_overlap get_new_batch_prefill get_next_batch_to_run
                    get_num_allocatable_reqs init_all_backends init_chunked_prefill
                    init_memory_pools init_model_worker init_moe_gemm_config init_overlap
                    init_schedule_policy init_target_memory_pool init_tp_model_worker
                    is_disable_overlap_for_batch record_batch_in_overlap run_batch
                    stash_chunked_request update_running_batch
                """.split()
            ]
            + ["sglang/srt/managers/scheduler.py::dispatch_event_loop"]
        )
    ),
    allowed_suffixes=tuple(sorted(".c .cc .cpp .cu .cuh .h .hpp .py".split())),
    forbidden_path_markers=tuple(sorted("""
        api detokenizer entrypoint grpc http logit logits logprob logprobs metric
        metrics profiler profiling result results sampler sampling timer timers timing tokenizer
    """.split())),
    forbidden_added_source=tuple(sorted("""
        sglang.srt.entrypoints sglang.srt.layers.logits_processor
        sglang.srt.layers.sampler sglang.srt.managers.detokenizer
        sglang.srt.managers.io_struct sglang.srt.managers.tokenizer
        sglang.srt.openai_api sglang.srt.sampling time.monotonic time.perf_counter
    """.split())),
)


@dataclass(frozen=True)
class DiscoveryBuildProfile:
    profile_id: str
    sglang_version: str
    arena: str
    model: str
    architecture: str
    tensor_parallel_size: int
    features: tuple[str, ...]
    build_inputs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _identifier(self.profile_id, field="profile_id"))
        _require(
            isinstance(self.sglang_version, str) and bool(self.sglang_version),
            "build profile sglang_version must not be empty",
        )
        object.__setattr__(self, "arena", _identifier(self.arena, field="build profile arena"))
        object.__setattr__(self, "model", _selector(self.model, field="build profile model"))
        if not isinstance(self.architecture, str) or _ARCH_RE.fullmatch(self.architecture) is None:
            raise DiscoveryError("build profile architecture must be canonical SM syntax")
        if type(self.tensor_parallel_size) is not int or not 1 <= self.tensor_parallel_size <= 65_536:
            raise DiscoveryError("build profile tensor_parallel_size is invalid")
        object.__setattr__(
            self, "features", _sorted_unique_strings(self.features, field="build profile features")
        )
        inputs = tuple(self.build_inputs)
        if not inputs or any(
            not isinstance(row, (list, tuple)) or len(row) != 2 for row in inputs
        ):
            raise DiscoveryError("build profile inputs must be name-sorted and unique")
        checked: list[tuple[str, str]] = []
        for name, digest in inputs:
            checked.append(
                (_identifier(name, field="build input name"), _digest(digest, field=f"build input {name}"))
            )
        if tuple(checked) != tuple(sorted(set(checked))) or len(
            {name for name, _digest_value in checked}
        ) != len(checked):
            raise DiscoveryError("build profile inputs must be name-sorted and unique")
        object.__setattr__(self, "build_inputs", tuple(checked))

    def to_dict(self) -> dict[str, object]:
        return {
            "architecture": self.architecture,
            "arena": self.arena,
            "build_inputs": [
                {"digest": digest, "name": name} for name, digest in self.build_inputs
            ],
            "features": list(self.features),
            "model": self.model,
            "profile_id": self.profile_id,
            "sglang_version": self.sglang_version,
            "tensor_parallel_size": self.tensor_parallel_size,
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.discovery.build-profile", self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryBuildProfile":
        row = _strict_object(
            value, frozenset(cls.__dataclass_fields__), name="discovery build profile"
        )
        inputs = row["build_inputs"]
        if not isinstance(inputs, list):
            raise DiscoveryError("discovery build inputs must be an array")
        payload = dict(row)
        payload["build_inputs"] = tuple(
            (
                item["name"],
                item["digest"],
            )
            for item in (
                _strict_object(
                    value,
                    frozenset({"digest", "name"}),
                    name="discovery build input",
                )
                for value in inputs
            )
        )
        try:
            result = cls(**payload)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise DiscoveryError(f"discovery build profile is invalid: {exc}") from None
        if result.to_dict() != dict(row):
            raise DiscoveryError("discovery build-profile serialization is not canonical")
        return result


def require_discovery_build_profile(
    manifest: DiscoveryManifest, profile: DiscoveryBuildProfile, policy: DiscoveryPolicy
) -> None:
    """Resolve proposal claims against one validator-registered build profile."""

    if manifest.build_profile != profile.profile_id:
        raise DiscoveryError("proposal requests another validator build profile")
    if profile.sglang_version != policy.sglang_version:
        raise DiscoveryError("build profile and discovery policy pin different SGLang versions")
    app = manifest.applicability
    if (
        profile.arena not in app.arenas
        or profile.model not in app.models
        or profile.architecture not in app.architectures
        or profile.tensor_parallel_size not in app.tensor_parallel_sizes
    ):
        raise DiscoveryError("registered build profile is outside proposal applicability")
    missing = set(manifest.dependencies) - set(profile.features)
    conflicts = set(manifest.conflicts) & set(profile.features)
    if missing:
        raise DiscoveryError(f"build profile lacks proposal dependencies: {tuple(sorted(missing))!r}")
    if conflicts:
        raise DiscoveryError(f"build profile activates proposal conflicts: {tuple(sorted(conflicts))!r}")


def _path_tokens(path: str) -> set[str]:
    return {
        token
        for component in PurePosixPath(path).parts
        for token in re.split(r"[^a-z0-9]+", component.lower())
        if token
    }


def validate_discovery_patch_path(policy: DiscoveryPolicy, path: str) -> None:
    path = _canonical_relative(path, field="SGLang patch path")
    pure = PurePosixPath(path)
    if pure.parts[:1] != ("sglang",):
        raise DiscoveryError(f"patch path is not rooted at sglang/: {path!r}")
    if pure.suffix not in policy.allowed_suffixes:
        raise DiscoveryError(f"patch path is not inspectable source: {path!r}")
    if path not in policy.allowed_files and not any(
        path.startswith(prefix) for prefix in policy.allowed_prefixes
    ):
        raise DiscoveryError(f"patch path is outside the validator-owned inference region: {path!r}")
    markers = _path_tokens(path) & set(policy.forbidden_path_markers)
    if markers:
        raise DiscoveryError(
            f"patch path enters excluded semantic surfaces {tuple(sorted(markers))!r}: {path!r}"
        )


def _symbol_names(policy: DiscoveryPolicy, path: str) -> tuple[str, ...]:
    prefix = path + "::"
    return tuple(row[len(prefix) :] for row in policy.allowed_symbol_regions if row.startswith(prefix))


def _symbol_ranges(source: str, wanted: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise DiscoveryError(f"pinned semantic-region source is invalid Python: {exc}") from None
    found: dict[str, tuple[int, int]] = {}

    def walk(nodes: list[ast.stmt], prefix: str = "") -> None:
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([node.lineno, *(row.lineno for row in node.decorator_list)])
                found[prefix + node.name] = (start, int(node.end_lineno or node.lineno))
            elif isinstance(node, ast.ClassDef):
                walk(node.body, prefix + node.name + ".")

    walk(tree.body)
    selected = {name: found[name] for name in wanted if name in found}
    if not selected:
        raise DiscoveryError(
            f"pinned source contains none of the allowed semantic regions {wanted!r}"
        )
    return selected


def _outside_symbol_projection(source: str, allowed: frozenset[str]) -> tuple[object, ...]:
    """AST identity with validator-admitted function bodies replaced by sentinels."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise DiscoveryError(f"semantic-region source is invalid Python: {exc}") from None

    def project(nodes: list[ast.stmt], prefix: str = "") -> tuple[object, ...]:
        rows: list[object] = []
        for node in nodes:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = prefix + node.name
                rows.append(
                    ("allowed", name)
                    if name in allowed
                    else ("node", ast.dump(node, include_attributes=False))
                )
            elif isinstance(node, ast.ClassDef):
                header = (
                    tuple(ast.dump(value, include_attributes=False) for value in node.bases),
                    tuple(ast.dump(value, include_attributes=False) for value in node.keywords),
                    tuple(ast.dump(value, include_attributes=False) for value in node.decorator_list),
                )
                rows.append(("class", node.name, header, project(node.body, prefix + node.name + ".")))
            else:
                rows.append(("node", ast.dump(node, include_attributes=False)))
        return tuple(rows)

    return project(tree.body)


def _validate_symbol_changes(
    file_patch: FilePatch, *, original_source: str, allowed_symbols: tuple[str, ...]
) -> None:
    old_ranges = tuple(_symbol_ranges(original_source, allowed_symbols).values())
    try:
        new_source = apply_file_patch(original_source, file_patch)
    except DepPatchError as exc:
        raise DiscoveryError(str(exc)) from None
    new_ranges = tuple(_symbol_ranges(new_source, allowed_symbols).values())
    allowed = frozenset(allowed_symbols)
    if _outside_symbol_projection(original_source, allowed) != _outside_symbol_projection(
        new_source, allowed
    ):
        raise DiscoveryError(
            f"patch {file_patch.path!r} changes Python semantics outside allowed symbols"
        )
    permitted = lambda line, ranges: any(start <= line <= end for start, end in ranges)
    for hunk in file_patch.hunks:
        old_line, new_line = hunk.old_start, hunk.new_start
        for change in hunk.lines:
            if change.startswith(" "):
                if not permitted(old_line, old_ranges) and permitted(new_line, new_ranges):
                    raise DiscoveryError(
                        f"patch {file_patch.path!r} absorbs unchanged line {old_line} "
                        "from outside its allowed symbols"
                    )
                old_line += 1
                new_line += 1
            elif change.startswith("-"):
                if not permitted(old_line, old_ranges):
                    raise DiscoveryError(
                        f"patch {file_patch.path!r} changes line {old_line} outside allowed symbols"
                    )
                old_line += 1
            elif change.startswith("+"):
                if not permitted(new_line, new_ranges):
                    raise DiscoveryError(
                        f"patch {file_patch.path!r} inserts at line {new_line} outside allowed symbols"
                    )
                new_line += 1


_DYNAMIC_SOURCE_NAMES = frozenset(
    {
        "__builtins__", "__dict__", "__import__", "builtins", "delattr",
        "eval", "exec", "getattr", "globals", "importlib", "locals", "monotonic",
        "openai_api", "perf_counter", "popen", "requests", "setattr", "socket",
        "subprocess", "system", "urllib", "vars",
    }
)


def _sensitive_import_aliases(source: str, policy: DiscoveryPolicy) -> set[str]:
    """Names through which added code could reuse an already-imported control surface."""

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise DiscoveryError(f"patched Python is invalid: {exc}") from None
    forbidden = tuple(value.lower() for value in policy.forbidden_added_source)
    markers = set(policy.forbidden_path_markers)
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.lower()
                if (
                    any(value in module for value in forbidden)
                    or _path_tokens(module.replace(".", "/")) & markers
                    or module.split(".")[0]
                    in {"importlib", "requests", "socket", "subprocess", "urllib"}
                ):
                    aliases.add((alias.asname or alias.name.split(".")[0]).lower())
        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").lower()
            for alias in node.names:
                imported = f"{module}.{alias.name.lower()}".strip(".")
                if (
                    any(value in imported for value in forbidden)
                    or _path_tokens(imported.replace(".", "/")) & markers
                    or imported.split(".")[0]
                    in {"importlib", "requests", "socket", "subprocess", "urllib"}
                    or imported in {"sys.modules", "os.popen", "os.system"}
                ):
                    aliases.add((alias.asname or alias.name).lower())
    return aliases


def _validate_added_source_names(
    policy: DiscoveryPolicy, *, original_source: str | None, new_source: str, added: str
) -> None:
    """Defense-in-depth against direct, relative, aliased, and dynamic control imports."""

    forbidden = set(_DYNAMIC_SOURCE_NAMES)
    forbidden.update(
        value.rsplit(".", 1)[-1].lower()
        for value in policy.forbidden_added_source
    )
    if original_source is not None:
        forbidden.update(_sensitive_import_aliases(original_source, policy))
    # Parsing the complete result catches relative imports whose added fragment is
    # not independently parseable.  Alias extraction catches reuse of an excluded
    # import that existed before the patch.
    forbidden.update(_sensitive_import_aliases(new_source, policy))
    added_names = set(re.findall(r"(?<![A-Za-z0-9_])[A-Za-z_][A-Za-z0-9_]*(?![A-Za-z0-9_])", added))
    overlap = tuple(sorted(forbidden & added_names))
    if overlap or re.search(r"\bsys\s*\.\s*modules\b", added):
        raise DiscoveryError(
            "patch added references to excluded/dynamic source names "
            f"{overlap or ('sys.modules',)!r}"
        )


def validate_discovery_file_patch(
    policy: DiscoveryPolicy,
    file_patch: FilePatch,
    *,
    original_source: str | None,
) -> None:
    validate_discovery_patch_path(policy, file_patch.path)
    symbols = _symbol_names(policy, file_patch.path)
    if symbols:
        if file_patch.is_new_file or original_source is None:
            raise DiscoveryError(
                f"patch {file_patch.path!r} requires pinned source for semantic-region validation"
            )
        _validate_symbol_changes(
            file_patch, original_source=original_source, allowed_symbols=symbols
        )
    added = "\n".join(
        line[1:]
        for hunk in file_patch.hunks
        for line in hunk.lines
        if line.startswith("+")
    ).lower()
    for forbidden in policy.forbidden_added_source:
        if forbidden.lower() in added:
            raise DiscoveryError(
                f"patch {file_patch.path!r} references excluded source {forbidden!r}"
            )
    if PurePosixPath(file_patch.path).suffix == ".py":
        try:
            new_source = apply_file_patch(original_source, file_patch)
        except DepPatchError as exc:
            raise DiscoveryError(str(exc)) from None
        _validate_added_source_names(
            policy,
            original_source=original_source,
            new_source=new_source,
            added=added,
        )


def _tree_inventory(
    root: Path,
    *,
    ignore_source_noise: bool = False,
) -> tuple[DiscoveryFile, ...]:
    root = _source_root(root, field="SGLang tree")
    rows: list[DiscoveryFile] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(
            name
            for name in directories
            if not (ignore_source_noise and name in _IGNORED_DIRECTORIES)
        )
        for name in directories:
            child = current_path / name
            info = child.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise DiscoveryError(f"SGLang tree contains unsafe directory entry: {child}")
        for name in sorted(files):
            child = current_path / name
            relative = child.relative_to(root)
            if ignore_source_noise and (
                name in _IGNORED_FILES
                or child.suffix in _IGNORED_SUFFIXES
                or name.startswith("._")
            ):
                continue
            logical = _canonical_relative(relative.as_posix(), field="SGLang tree path")
            raw = _stable_read(child, field=f"SGLang tree file {logical!r}")
            rows.append(DiscoveryFile(logical, sha256_hex(raw), len(raw)))
    rows.sort(key=lambda row: row.path)
    if not rows:
        raise DiscoveryError("SGLang tree is empty")
    return tuple(rows)


def discovery_tree_digest(files: tuple[DiscoveryFile, ...]) -> str:
    return canonical_digest(
        "optima.discovery.sglang-tree", {"files": [row.to_dict() for row in files]}
    )


@dataclass(frozen=True)
class ValidatedDiscoveryPatchSet:
    proposal_digest: str
    policy_digest: str
    stock_site_root: Path
    stock_files: tuple[DiscoveryFile, ...]
    parsed: tuple[tuple[str, tuple[FilePatch, ...]], ...]
    touched_paths: tuple[str, ...]

    @property
    def stock_tree_digest(self) -> str:
        return discovery_tree_digest(self.stock_files)


def validate_discovery_patch_set(
    discovery: InspectedDiscovery,
    policy: DiscoveryPolicy,
    stock_site_root: str | Path,
) -> ValidatedDiscoveryPatchSet:
    """Parse and policy-check frozen patches against one exact pinned tree."""

    if not isinstance(discovery, InspectedDiscovery) or not isinstance(policy, DiscoveryPolicy):
        raise DiscoveryError("patch validation requires typed discovery and policy values")
    site = _source_root(stock_site_root, field="stock SGLang site root")
    package = site / "sglang"
    stock_files = _tree_inventory(package, ignore_source_noise=True)
    parsed: list[tuple[str, tuple[FilePatch, ...]]] = []
    touched: set[str] = set()
    for relative, text in discovery.patch_texts:
        try:
            file_patches = parse_patch_text(text)
        except DepPatchError as exc:  # frozen values are still independently reopened
            raise DiscoveryError(f"patch {relative!r} rejected: {exc}") from None
        for file_patch in file_patches:
            pinned = site.joinpath(*PurePosixPath(file_patch.path).parts)
            original: str | None = None
            if pinned.exists():
                raw = _stable_read(pinned, field=f"pinned source {file_patch.path!r}")
                try:
                    original = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise DiscoveryError(
                        f"pinned source {file_patch.path!r} is not UTF-8: {exc}"
                    ) from None
            validate_discovery_file_patch(
                policy, file_patch, original_source=original
            )
            if file_patch.path in touched:
                raise DiscoveryError(
                    f"SGLang file {file_patch.path!r} is touched more than once"
                )
            touched.add(file_patch.path)
        parsed.append((relative, file_patches))
    return ValidatedDiscoveryPatchSet(
        proposal_digest=discovery.proposal_digest,
        policy_digest=policy.digest,
        stock_site_root=site,
        stock_files=stock_files,
        parsed=tuple(parsed),
        touched_paths=tuple(sorted(touched)),
    )


@dataclass(frozen=True)
class DiscoveryOverlayIdentity:
    proposal_digest: str
    policy_digest: str
    build_profile_digest: str
    stock_tree_digest: str
    result_tree_digest: str
    files: tuple[DiscoveryFile, ...]
    touched_files: tuple[DiscoveryFile, ...]

    def __post_init__(self) -> None:
        for field in (
            "proposal_digest", "policy_digest", "build_profile_digest",
            "stock_tree_digest", "result_tree_digest",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        files = tuple(self.files)
        touched = tuple(self.touched_files)
        for name, rows in (("files", files), ("touched_files", touched)):
            if any(type(row) is not DiscoveryFile for row in rows):
                raise DiscoveryError(f"overlay {name} must contain DiscoveryFile values")
            paths = tuple(row.path for row in rows)
            if paths != tuple(sorted(set(paths))):
                raise DiscoveryError(f"overlay {name} must be path-sorted and unique")
        if not files or not set(touched).issubset(set(files)):
            raise DiscoveryError("overlay touched-file inventory is not a subset")
        if discovery_tree_digest(files) != self.result_tree_digest:
            raise DiscoveryError("overlay result_tree_digest differs from file inventory")
        object.__setattr__(self, "files", files)
        object.__setattr__(self, "touched_files", touched)

    def to_dict(self) -> dict[str, object]:
        return {
            "build_profile_digest": self.build_profile_digest,
            "files": [row.to_dict() for row in self.files],
            "policy_digest": self.policy_digest,
            "proposal_digest": self.proposal_digest,
            "result_tree_digest": self.result_tree_digest,
            "stock_tree_digest": self.stock_tree_digest,
            "touched_files": [row.to_dict() for row in self.touched_files],
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.discovery.overlay-identity", self.to_dict())

    @classmethod
    def from_dict(cls, value: object) -> "DiscoveryOverlayIdentity":
        fields = frozenset(
            {
                "build_profile_digest", "files", "policy_digest", "proposal_digest",
                "result_tree_digest", "stock_tree_digest", "touched_files",
            }
        )
        row = _strict_object(value, fields, name="discovery overlay identity")
        if not isinstance(row["files"], list) or not isinstance(row["touched_files"], list):
            raise DiscoveryError("overlay identity inventories must be arrays")
        return cls(
            proposal_digest=row["proposal_digest"],  # type: ignore[arg-type]
            policy_digest=row["policy_digest"],  # type: ignore[arg-type]
            build_profile_digest=row["build_profile_digest"],  # type: ignore[arg-type]
            stock_tree_digest=row["stock_tree_digest"],  # type: ignore[arg-type]
            result_tree_digest=row["result_tree_digest"],  # type: ignore[arg-type]
            files=tuple(DiscoveryFile.from_dict(item) for item in row["files"]),
            touched_files=tuple(
                DiscoveryFile.from_dict(item) for item in row["touched_files"]
            ),
        )


@dataclass(frozen=True)
class MaterializedDiscoveryOverlay:
    root: Path
    site_root: Path
    identity: DiscoveryOverlayIdentity

    @property
    def identity_digest(self) -> str:
        return self.identity.digest


@dataclass(frozen=True)
class DiscoveryEngineBinding:
    materialized_tree: MaterializedEngineTree
    discovery: InspectedDiscovery
    policy: DiscoveryPolicy
    build_profile: DiscoveryBuildProfile
    incumbent_stack_digest: str
    incumbent_tree_digest: str

    def __post_init__(self) -> None:
        from optima.engine_tree import MaterializedEngineTree

        if type(self.materialized_tree) is not MaterializedEngineTree:
            raise DiscoveryError("discovery binding requires a materialized engine tree")
        if type(self.discovery) is not InspectedDiscovery:
            raise DiscoveryError("discovery binding proposal is not typed")
        if type(self.policy) is not DiscoveryPolicy:
            raise DiscoveryError("discovery binding policy is not typed")
        if type(self.build_profile) is not DiscoveryBuildProfile:
            raise DiscoveryError("discovery binding build profile is not typed")
        for field in ("incumbent_stack_digest", "incumbent_tree_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        require_discovery_build_profile(
            self.discovery.manifest, self.build_profile, self.policy
        )
        expected = discovery_candidate_stack_digest(
            incumbent_stack_digest=self.incumbent_stack_digest,
            incumbent_tree_digest=self.incumbent_tree_digest,
            proposal_digest=self.discovery.proposal_digest,
            policy_digest=self.policy.digest,
            build_profile_digest=self.build_profile.digest,
        )
        if self.materialized_tree.stack_digest != expected:
            raise DiscoveryError("discovery engine tree has another source-intent stack")
        if self.materialized_tree.tree_digest == self.incumbent_tree_digest:
            raise DiscoveryError("discovery engine tree does not differ from its incumbent")


def reopen_discovery_engine_binding(
    materialized_tree: MaterializedEngineTree,
) -> DiscoveryEngineBinding:
    """Reopen the validator projection consumed by the fixed OCI builder."""

    from optima.engine_tree import MaterializedEngineTree, reopen_materialized_engine_tree

    if type(materialized_tree) is not MaterializedEngineTree:
        raise DiscoveryError("discovery engine binding requires a materialized tree")
    try:
        reopened = reopen_materialized_engine_tree(
            materialized_tree.root, expected_tree_digest=materialized_tree.tree_digest
        )
    except (OSError, TypeError, ValueError) as exc:
        raise DiscoveryError(f"discovery engine tree failed to reopen: {exc}") from None
    if reopened != materialized_tree:
        raise DiscoveryError("discovery engine tree changed before binding")

    raw = _stable_read(
        reopened.root / _DISCOVERY_ENGINE_METADATA,
        field="discovery engine metadata",
        limit=32 << 20,
    )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise DiscoveryError(f"discovery engine metadata is invalid: {exc}") from None
    metadata = _strict_object(
        value, _DISCOVERY_ENGINE_FIELDS, name="discovery engine metadata"
    )
    if metadata["schema"] != _DISCOVERY_ENGINE_SCHEMA:
        raise DiscoveryError("discovery engine metadata schema is unsupported")
    canonical = json.dumps(
        metadata, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    if raw != canonical:
        raise DiscoveryError("discovery engine metadata is not canonical")

    policy = DiscoveryPolicy.from_dict(metadata["policy"])
    build_profile = DiscoveryBuildProfile.from_dict(metadata["build_profile"])
    if policy.digest != metadata["policy_digest"]:
        raise DiscoveryError("discovery engine policy digest mismatch")
    if build_profile.digest != metadata["build_profile_digest"]:
        raise DiscoveryError("discovery engine build-profile digest mismatch")
    proposal_rows = metadata["proposal_files"]
    if not isinstance(proposal_rows, list):
        raise DiscoveryError("discovery engine proposal inventory must be an array")
    proposal_files = tuple(DiscoveryFile.from_dict(row) for row in proposal_rows)
    discovery = inspect_discovery(reopened.root / "discovery")
    if proposal_files != discovery.files:
        raise DiscoveryError("discovery engine proposal inventory mismatch")
    if metadata["proposal_digest"] != discovery.proposal_digest:
        raise DiscoveryError("discovery engine proposal digest mismatch")
    return DiscoveryEngineBinding(
        materialized_tree=reopened,
        discovery=discovery,
        policy=policy,
        build_profile=build_profile,
        incumbent_stack_digest=metadata["incumbent_stack_digest"],  # type: ignore[arg-type]
        incumbent_tree_digest=metadata["incumbent_tree_digest"],  # type: ignore[arg-type]
    )


def _copy_inventory(source: Path, destination: Path, files: tuple[DiscoveryFile, ...]) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for row in files:
        source_file = source.joinpath(*PurePosixPath(row.path).parts)
        raw = _stable_read(source_file, field=f"stock source {row.path!r}")
        if len(raw) != row.size or sha256_hex(raw) != row.sha256:
            raise DiscoveryError(f"stock SGLang tree changed while copying {row.path!r}")
        output = destination.joinpath(*PurePosixPath(row.path).parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
    if (
        _tree_inventory(source, ignore_source_noise=True) != files
        or _tree_inventory(destination) != files
    ):
        raise DiscoveryError("stock SGLang tree changed during overlay copy")


def build_discovery_overlay_stage(
    discovery: InspectedDiscovery,
    *,
    stock_site_root: str | Path,
    native_stage_root: str | Path,
    policy: DiscoveryPolicy,
    build_profile: DiscoveryBuildProfile,
) -> DiscoveryOverlayIdentity:
    """Build the fixed discovery envelope inside a lease-owned native stage.

    Publication, collision handling, immutable modes, resource bounds, and full
    inventory reopening belong to :mod:`optima.eval.native_artifact`.  This
    fixed worker operation only copies the image-pinned SGLang package, applies
    already-policy-checked text patches, and writes discovery's semantic stamp.
    """

    if type(discovery) is not InspectedDiscovery:
        raise DiscoveryError("stage build requires a trusted InspectedDiscovery")
    require_discovery_build_profile(discovery.manifest, build_profile, policy)
    validated = validate_discovery_patch_set(discovery, policy, stock_site_root)
    stage = _source_root(native_stage_root, field="native artifact stage")
    overlays = stage / "dep_overlays"
    if os.path.lexists(overlays):
        info = overlays.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise DiscoveryError("native artifact dep_overlays entry is unsafe")
    else:
        overlays.mkdir()
    destination = overlays / "discovery"
    if os.path.lexists(destination):
        raise DiscoveryError("discovery native stage envelope already exists")

    site = destination / "site"
    package = site / "sglang"
    destination.mkdir()
    site.mkdir()
    _copy_inventory(
        validated.stock_site_root / "sglang", package, validated.stock_files
    )
    touched: list[DiscoveryFile] = []
    for _relative, file_patches in validated.parsed:
        for file_patch in file_patches:
            target = site.joinpath(*PurePosixPath(file_patch.path).parts)
            original: str | None = None
            if target.exists():
                try:
                    original = _stable_read(
                        target, field=f"overlay target {file_patch.path!r}"
                    ).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise DiscoveryError(
                        f"overlay target {file_patch.path!r} is not UTF-8: {exc}"
                    ) from None
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
            try:
                patched = apply_file_patch(original, file_patch)
            except DepPatchError as exc:
                raise DiscoveryError(str(exc)) from None
            if target.suffix == ".py":
                try:
                    compile(patched, file_patch.path, "exec")
                except SyntaxError as exc:
                    raise DiscoveryError(
                        f"patched Python is invalid: {file_patch.path}: {exc}"
                    ) from None
            target.write_text(patched, encoding="utf-8")

    files = _tree_inventory(package)
    by_path = {row.path: row for row in files}
    for path in validated.touched_paths:
        relative = PurePosixPath(path).relative_to("sglang").as_posix()
        row = by_path.get(relative)
        if row is None:
            raise DiscoveryError(f"patched output is missing: {path!r}")
        touched.append(row)
    identity = DiscoveryOverlayIdentity(
        proposal_digest=discovery.proposal_digest,
        policy_digest=policy.digest,
        build_profile_digest=build_profile.digest,
        stock_tree_digest=validated.stock_tree_digest,
        result_tree_digest=discovery_tree_digest(files),
        files=files,
        touched_files=tuple(touched),
    )
    stamp = {"identity": identity.to_dict(), "schema": DISCOVERY_OVERLAY_SCHEMA}
    (destination / "overlay.json").write_bytes(canonical_json_bytes(stamp) + b"\n")
    return identity


def _mount_is_read_only(path: Path) -> bool:
    try:
        return bool(os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1))
    except OSError:
        return False


def reopen_discovery_overlay(
    publication: NativeArtifactPublication,
    *,
    expected_identity_digest: str,
    require_read_only: bool = False,
    read_only_check: Callable[[Path], bool] = _mount_is_read_only,
) -> MaterializedDiscoveryOverlay:
    """Project a discovery overlay from one completely reopened publication."""

    if type(publication) is not NativeArtifactPublication:
        raise DiscoveryError("discovery reopen requires a native artifact publication")
    expected = _digest(expected_identity_digest, field="expected overlay identity digest")
    envelope_rows = tuple(
        row
        for row in publication.files
        if row.path.startswith(_DISCOVERY_ENVELOPE + "/")
    )
    stamp_rows = tuple(row for row in envelope_rows if row.path == _DISCOVERY_STAMP)
    package_rows = tuple(
        row for row in envelope_rows if row.path.startswith(_DISCOVERY_PACKAGE)
    )
    if (
        len(stamp_rows) != 1
        or not package_rows
        or len(envelope_rows) != len(package_rows) + 1
    ):
        raise DiscoveryError("discovery overlay publication envelope is not closed")

    root = _source_root(
        publication.root / "dep_overlays" / "discovery",
        field="discovery overlay",
    )
    site = _source_root(root / "site", field="discovery overlay site")
    _source_root(site / "sglang", field="discovery overlay package")
    raw = _stable_read(root / "overlay.json", field="overlay stamp", limit=32 << 20)
    stamp_row = stamp_rows[0]
    if len(raw) != stamp_row.size or sha256_hex(raw) != stamp_row.sha256:
        raise DiscoveryError("discovery overlay stamp differs from native inventory")
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise DiscoveryError(f"discovery overlay stamp is invalid: {exc}") from None
    if raw != canonical_json_bytes(value) + b"\n":
        raise DiscoveryError("discovery overlay stamp is not canonical")
    stamp = _strict_object(value, _STAMP_FIELDS, name="discovery overlay stamp")
    if stamp["schema"] != DISCOVERY_OVERLAY_SCHEMA:
        raise DiscoveryError("discovery overlay stamp schema is unsupported")
    identity = DiscoveryOverlayIdentity.from_dict(stamp["identity"])
    if identity.digest != expected:
        raise DiscoveryError("discovery overlay identity digest mismatch")
    files = tuple(
        DiscoveryFile(
            row.path[len(_DISCOVERY_PACKAGE) :], row.sha256, row.size
        )
        for row in package_rows
    )
    if files != identity.files:
        raise DiscoveryError("discovery overlay tree differs from its exact inventory")
    by_path = {row.path: row for row in files}
    if any(by_path.get(row.path) != row for row in identity.touched_files):
        raise DiscoveryError("discovery overlay touched-file inventory differs")
    if require_read_only and not read_only_check(publication.root):
        raise DiscoveryError("discovery overlay is not mounted read-only")
    return MaterializedDiscoveryOverlay(root=root, site_root=site, identity=identity)


def discovery_selected_delta_digest(
    *,
    proposal_digest: str,
    policy_digest: str,
    build_profile_digest: str,
) -> str:
    """Source intent of one discovery delta, before build output exists."""

    return canonical_digest(
        "optima.discovery.selected-delta",
        {
            "build_profile_digest": _digest(
                build_profile_digest, field="build_profile_digest"
            ),
            "policy_digest": _digest(policy_digest, field="policy_digest"),
            "proposal_digest": _digest(proposal_digest, field="proposal_digest"),
        },
    )


def discovery_candidate_stack_digest(
    *,
    incumbent_stack_digest: str,
    incumbent_tree_digest: str,
    proposal_digest: str,
    policy_digest: str,
    build_profile_digest: str,
) -> str:
    """Pre-build identity of ephemeral C without a permanent stack entry."""

    return canonical_digest(
        "optima.discovery.candidate-stack",
        {
            "incumbent_stack_digest": _digest(incumbent_stack_digest, field="incumbent_stack_digest"),
            "incumbent_tree_digest": _digest(incumbent_tree_digest, field="incumbent_tree_digest"),
            "selected_delta_digest": discovery_selected_delta_digest(
                proposal_digest=proposal_digest,
                policy_digest=policy_digest,
                build_profile_digest=build_profile_digest,
            ),
        },
    )


@dataclass(frozen=True)
class DiscoveryArmPlan:
    incumbent: EvaluationStackManifest
    incumbent_tree_digest: str
    candidate_stack_digest: str
    candidate_tree_digest: str
    proposal_digest: str
    policy_digest: str
    build_profile_digest: str
    overlay_identity_digest: str

    def __post_init__(self) -> None:
        if type(self.incumbent) is not EvaluationStackManifest:
            raise DiscoveryError("discovery incumbent must be an EvaluationStackManifest")
        for field in self.__dataclass_fields__:
            if field == "incumbent":
                continue
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        expected = discovery_candidate_stack_digest(
            incumbent_stack_digest=self.incumbent.digest,
            incumbent_tree_digest=self.incumbent_tree_digest,
            proposal_digest=self.proposal_digest,
            policy_digest=self.policy_digest,
            build_profile_digest=self.build_profile_digest,
        )
        if self.candidate_stack_digest != expected:
            raise DiscoveryError("candidate stack digest is not the ephemeral discovery identity")
        if self.candidate_tree_digest == self.incumbent_tree_digest:
            raise DiscoveryError("discovery challenger tree must differ from incumbent")

    @classmethod
    def create(
        cls,
        *,
        incumbent: EvaluationStackManifest,
        incumbent_tree_digest: str,
        candidate_tree_digest: str,
        proposal_digest: str,
        policy_digest: str,
        build_profile_digest: str,
        overlay_identity_digest: str,
    ) -> "DiscoveryArmPlan":
        if type(incumbent) is not EvaluationStackManifest:
            raise DiscoveryError("discovery incumbent must be an EvaluationStackManifest")
        candidate_stack = discovery_candidate_stack_digest(
            incumbent_stack_digest=incumbent.digest,
            incumbent_tree_digest=incumbent_tree_digest,
            proposal_digest=proposal_digest,
            policy_digest=policy_digest,
            build_profile_digest=build_profile_digest,
        )
        return cls(
            incumbent, incumbent_tree_digest, candidate_stack,
            candidate_tree_digest, proposal_digest, policy_digest,
            build_profile_digest, overlay_identity_digest,
        )

    @property
    def incumbent_stack_digest(self) -> str:
        return self.incumbent.digest

    @property
    def baseline_before(self) -> StackArmIdentity:
        return StackArmIdentity(self.incumbent.digest, self.incumbent_tree_digest)

    @property
    def baseline_after(self) -> StackArmIdentity:
        return StackArmIdentity(self.incumbent.digest, self.incumbent_tree_digest)

    @property
    def challenger(self) -> StackArmIdentity:
        return StackArmIdentity(self.candidate_stack_digest, self.candidate_tree_digest)

    @property
    def selected_delta_digest(self) -> str:
        return discovery_selected_delta_digest(
            proposal_digest=self.proposal_digest,
            policy_digest=self.policy_digest,
            build_profile_digest=self.build_profile_digest,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_after": self.baseline_after.to_dict(),
            "baseline_before": self.baseline_before.to_dict(),
            "build_profile_digest": self.build_profile_digest,
            "challenger": self.challenger.to_dict(),
            "overlay_identity_digest": self.overlay_identity_digest,
            "policy_digest": self.policy_digest,
            "proposal_digest": self.proposal_digest,
            "schema_version": 1,
            "selected_delta_digest": self.selected_delta_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.discovery.arm-plan", self.to_dict())


@dataclass(frozen=True)
class DiscoveryWinRecord:
    arm_digest: str
    proposal_digest: str
    overlay_identity_digest: str
    qualification_digest: str
    requested_promotion: str

    def __post_init__(self) -> None:
        for field in ("arm_digest", "proposal_digest", "overlay_identity_digest", "qualification_digest"):
            object.__setattr__(self, field, _digest(getattr(self, field), field=field))
        if (
            not isinstance(self.requested_promotion, str)
            or self.requested_promotion not in DISCOVERY_PROMOTIONS
        ):
            raise DiscoveryError("win record requested_promotion is unsupported")

    def to_dict(self) -> dict[str, object]:
        return {
            "arm_digest": self.arm_digest,
            "decision": "pass",
            "overlay_identity_digest": self.overlay_identity_digest,
            "proposal_digest": self.proposal_digest,
            "qualification_digest": self.qualification_digest,
            "requested_promotion": self.requested_promotion,
            "schema_version": 1,
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.discovery.win-record", self.to_dict())


@dataclass(frozen=True)
class DiscoveryPromotion:
    win_record_digest: str
    disposition: str
    review_digest: str
    subject: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "win_record_digest", _digest(self.win_record_digest, field="win_record_digest")
        )
        object.__setattr__(self, "review_digest", _digest(self.review_digest, field="review_digest"))
        if not isinstance(self.disposition, str) or self.disposition not in DISCOVERY_PROMOTIONS:
            raise DiscoveryError("promotion disposition is unsupported")
        if self.disposition == "bounty_only":
            if self.subject is not None:
                raise DiscoveryError("bounty-only promotion cannot name a shipping subject")
        else:
            object.__setattr__(self, "subject", _identifier(self.subject, field="promotion subject"))

    def to_dict(self) -> dict[str, object]:
        return {
            "disposition": self.disposition,
            "review_digest": self.review_digest,
            "schema_version": 1,
            "subject": self.subject,
            "win_record_digest": self.win_record_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest("optima.discovery.promotion", self.to_dict())
