"""In-container worker for one isolated, open-ended engine session.

This module is intentionally safe to import in the trusted controller: its module
body imports only the standard library and the data-only session protocol.  Runtime,
candidate, native-artifact, and SGLang imports occur only after a live preflight has
been emitted on the dedicated controller pipe.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
import secrets
import stat
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from optima.eval.engine_worker import (
    _environment,
    _path_mount_is_read_only as _path_is_read_only,
)
from optima.eval.oci_session_protocol import (
    CONTROL_MAGIC,
    CONTAINER_MODEL_PATH,
    FRAME_HEADER_BYTES,
    MAX_BATCH_REQUEST_BYTES,
    MAX_CONTROL_BYTES,
    MAX_INIT_BYTES,
    BatchEvidence,
    BatchRequest,
    EngineSessionConfig,
    PromptEvidence,
    RuntimePreflightFacts,
    SessionProtocolError,
    decode_message,
    error_message,
    evidence_frame,
    frame_message,
    preflight_message,
    ready_message,
    validate_batch_request,
    validate_init,
    validate_preflight_accept,
)
from optima.seams import seam_binding_environment


CONTAINER_TREE_PATH = "/optima/engine-tree"
CONTAINER_ARTIFACT_BASE = "/optima/native-artifacts"
CONTAINER_CACHE_PATH = "/optima/runtime-cache"
DISCOVERY_OVERLAY_RELPATH = Path("dep_overlays/discovery")
NVIDIA_SMI = "/usr/bin/nvidia-smi"

_DIGEST_ENV = {
    "runtime_digest": "OPTIMA_RUNTIME_DIGEST",
    "worker_distribution_digest": "OPTIMA_WORKER_DISTRIBUTION_DIGEST",
    "model_revision_digest": "OPTIMA_MODEL_REVISION_DIGEST",
    "model_manifest_digest": "OPTIMA_MODEL_MANIFEST_DIGEST",
    "model_content_digest": "OPTIMA_MODEL_CONTENT_DIGEST",
}


class SessionWorkerError(RuntimeError):
    """The live isolated worker or engine violated its launch contract."""


def _read_exact(fd: int, size: int) -> bytes:
    if type(size) is not int or size < 0:
        raise SessionProtocolError("session read size is invalid")
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        try:
            chunk = os.read(fd, min(remaining, 1 << 20))
        except InterruptedError:
            continue
        if not chunk:
            raise SessionProtocolError("controller closed a partial session request")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_control_frame(fd: int, *, max_bytes: int) -> dict[str, Any]:
    header = _read_exact(fd, FRAME_HEADER_BYTES)
    if header[:4] != CONTROL_MAGIC:
        raise SessionProtocolError("controller frame magic/version mismatch")
    size = struct.unpack(">I", header[4:8])[0]
    if size > max_bytes:
        raise SessionProtocolError("controller frame exceeds its hard bound")
    return decode_message(_read_exact(fd, size), max_bytes=max_bytes)


def _write_all(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    offset = 0
    while offset < len(view):
        try:
            written = os.write(fd, view[offset:])
        except InterruptedError:
            continue
        if written <= 0:
            raise SessionProtocolError("worker could not write session evidence")
        offset += written


def _reserve_protocol_fd() -> int:
    """Keep original stdout as CLOEXEC protocol and silence the engine tree."""

    protocol = os.dup(1)
    os.set_inheritable(protocol, False)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    finally:
        os.close(devnull)
    return protocol


def _reserve_control_fd(input_fd: int) -> int:
    """Keep controller input CLOEXEC and hide it from spawned engine children."""

    control = os.dup(input_fd)
    os.set_inheritable(control, False)
    if input_fd == 0:
        devnull = os.open(os.devnull, os.O_RDONLY)
        try:
            os.dup2(devnull, 0)
        finally:
            os.close(devnull)
    return control


def _required_digest_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if (
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise SessionWorkerError(f"{name} must be a nonzero lowercase SHA-256 digest")
    return value


def _required_sglang_version() -> str:
    value = os.environ.get("OPTIMA_EXPECTED_SGLANG_VERSION", "").strip()
    if (
        not value
        or len(value) > 256
        or any(character.isspace() or character == "\x00" for character in value)
    ):
        raise SessionWorkerError("expected SGLang version is invalid")
    return value


def _read_only_directory(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    return (
        not stat.S_ISLNK(info.st_mode)
        and stat.S_ISDIR(info.st_mode)
        and _path_is_read_only(path)
    )


def _validate_private_cache() -> None:
    path = Path(CONTAINER_CACHE_PATH)
    try:
        info = path.lstat()
    except OSError as exc:
        raise SessionWorkerError(f"private runtime cache is unavailable: {exc}") from None
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or info.st_uid != os.geteuid()
        or info.st_gid != os.getegid()
        or _path_is_read_only(path)
        or not os.path.ismount(path)
    ):
        raise SessionWorkerError("private runtime cache is not an owned writable 0700 mount")
    probe = path / (".preflight-" + secrets.token_hex(16))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = -1
    try:
        fd = os.open(probe, flags, 0o600)
        os.write(fd, b"preflight")
        os.fsync(fd)
    except OSError as exc:
        raise SessionWorkerError(f"private runtime cache write probe failed: {exc}") from None
    finally:
        if fd >= 0:
            os.close(fd)
        probe.unlink(missing_ok=True)


def _run_nvidia_smi(*args: str) -> str:
    try:
        result = subprocess.run(
            (NVIDIA_SMI, *args),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=30,
            check=False,
            close_fds=True,
            shell=False,
            env={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
    except (OSError, UnicodeError, subprocess.TimeoutExpired) as exc:
        raise SessionWorkerError(f"live GPU preflight failed: {exc}") from None
    if result.returncode != 0 or result.stderr.strip():
        raise SessionWorkerError(
            f"live GPU preflight exited {result.returncode}: {result.stderr[:512]}"
        )
    if len(result.stdout.encode("utf-8")) > 4 << 20:
        raise SessionWorkerError("live GPU preflight output exceeds its hard bound")
    return result.stdout


def _gpu_architectures() -> tuple[str, ...]:
    text = _run_nvidia_smi(
        "--query-gpu=compute_cap", "--format=csv,noheader,nounits"
    )
    architectures: list[str] = []
    for raw in text.splitlines():
        value = raw.strip()
        pieces = value.split(".")
        if (
            len(pieces) != 2
            or not all(piece.isdigit() for piece in pieces)
            or not 1 <= len(pieces[0]) <= 2
            or len(pieces[1]) != 1
        ):
            raise SessionWorkerError(f"invalid live GPU compute capability {value!r}")
        architectures.append(f"sm{pieces[0]}{pieces[1]}")
    if not 1 <= len(architectures) <= 64:
        raise SessionWorkerError("live GPU preflight returned an invalid device count")
    return tuple(architectures)


def _topology_digest() -> str:
    """Digest the visible GPU topology square using the host provisioning schema."""

    clean = _run_nvidia_smi("topo", "-m")
    # nvidia-smi underlines the topology header even without a TTY on some
    # drivers. Normalize only its known SGR pair and reject every other escape.
    clean = clean.replace("\x1b[4m", "").replace("\x1b[0m", "")
    if "\x1b" in clean:
        raise SessionWorkerError("live GPU topology contains an unknown escape")
    rows = [line.split() for line in clean.splitlines() if line.strip()]
    header: list[str] | None = None
    for row in rows:
        run: list[str] = []
        for cell in row:
            if cell.startswith("GPU") and cell[3:].isdigit():
                run.append(cell)
            elif run:
                break
        if run:
            header = run
            break
    if not header:
        raise SessionWorkerError("live GPU topology lacks a canonical header")
    links: dict[str, list[str]] = {}
    allowed = {"X", "PIX", "PXB", "PHB", "NODE", "SYS"}
    for row in rows:
        if row and row[0] in header and len(row) >= 1 + len(header):
            selected = row[1 : 1 + len(header)]
            if all(value in allowed or (value.startswith("NV") and value[2:].isdigit())
                   for value in selected):
                links[row[0]] = selected
    if set(links) != set(header):
        raise SessionWorkerError("live GPU topology square is incomplete")
    matrix = [[links[label][column] for column in range(len(header))] for label in header]
    if any(row[index] != "X" for index, row in enumerate(matrix)):
        raise SessionWorkerError("live GPU topology diagonal is malformed")
    payload = json.dumps(
        {"matrix": matrix, "schema": "optima-gpu-topology-v1"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _artifact_root() -> Path:
    raw = os.environ.get("OPTIMA_NATIVE_ARTIFACT_ROOT", "").strip()
    if not raw:
        raise SessionWorkerError("OPTIMA_NATIVE_ARTIFACT_ROOT is missing")
    requested = Path(raw)
    base = Path(CONTAINER_ARTIFACT_BASE)
    if (
        not requested.is_absolute()
        or requested != Path(os.path.normpath(requested))
        or requested == base
        or base not in requested.parents
    ):
        raise SessionWorkerError("native artifact root escapes its fixed container mount")
    return requested


def _requested_discovery_overlay(
    *, reference_mode: bool
) -> tuple[Path, str] | None:
    """Read the closed host-selected discovery binding before engine entry."""

    from optima import discovery_overlay

    armed = os.environ.get(discovery_overlay.ARMED, "").strip()
    values = {
        key: os.environ.get(key, "").strip()
        for key in discovery_overlay.DISCOVERY_ENVIRONMENT_KEYS
    }
    if not armed:
        if any(values.values()):
            raise SessionWorkerError(
                "disabled discovery launch contains an ambient marker"
            )
        return None
    if armed != "1" or reference_mode:
        raise SessionWorkerError(
            "discovery activation is invalid for this session role"
        )
    allowed = {
        discovery_overlay.ARMED,
        discovery_overlay.EXPECTED_IDENTITY,
    }
    if any(value for key, value in values.items() if key not in allowed):
        raise SessionWorkerError(
            "discovery launch contains a worker-owned transient marker"
        )
    identity = _required_digest_env(discovery_overlay.EXPECTED_IDENTITY)
    root = _artifact_root() / DISCOVERY_OVERLAY_RELPATH
    if not _read_only_directory(root):
        raise SessionWorkerError(
            "discovery overlay is absent from the read-only native publication"
        )
    return root, identity


def _validate_live_preflight(
    config: EngineSessionConfig, *, launch_digest: str
) -> tuple[RuntimePreflightFacts, object]:
    """Reopen every executable input and collect facts before candidate entry."""

    from optima.engine_tree import reopen_materialized_engine_tree
    from optima.eval.engine_worker import (
        _egress_is_blocked,
        _loopback_is_up,
        _network_namespace_is_loopback_only,
        _process_sandbox_is_hardened,
    )
    from optima.eval.native_artifact import reopen_native_artifact

    if os.environ.get("OPTIMA_LAUNCH_DIGEST", "").strip() != launch_digest:
        raise SessionWorkerError("live launch digest differs from the init binding")
    if _required_digest_env("OPTIMA_ENGINE_CONFIG_DIGEST") != config.digest:
        raise SessionWorkerError("live engine configuration differs from the init binding")
    expected_tree = _required_digest_env("OPTIMA_ENGINE_TREE_DIGEST")
    expected_stack = _required_digest_env("OPTIMA_STACK_DIGEST")
    tree = reopen_materialized_engine_tree(
        CONTAINER_TREE_PATH, expected_tree_digest=expected_tree
    )
    if tree.stack_digest != expected_stack:
        raise SessionWorkerError("materialized tree stack digest differs from launch identity")

    artifact = _artifact_root()
    reopen_native_artifact(
        artifact,
        expected_build_spec_digest=_required_digest_env(
            "OPTIMA_NATIVE_BUILD_SPEC_DIGEST"
        ),
        expected_publication_digest=_required_digest_env(
            "OPTIMA_NATIVE_ARTIFACT_PUBLICATION_DIGEST"
        ),
        expected_owner_uid=None,
    )
    inputs = (Path(CONTAINER_MODEL_PATH), Path(CONTAINER_TREE_PATH), artifact)
    if any(not _read_only_directory(path) for path in inputs):
        raise SessionWorkerError("model, engine tree, and native artifacts must be read-only")
    _validate_private_cache()
    if not (
        _loopback_is_up()
        and _network_namespace_is_loopback_only()
        and _egress_is_blocked()
        and _process_sandbox_is_hardened()
    ):
        raise SessionWorkerError("live OCI sandbox is not loopback-only and hardened")

    expected_sglang = _required_sglang_version()
    observed_sglang = importlib.metadata.version("sglang")
    if observed_sglang != expected_sglang:
        raise SessionWorkerError("installed SGLang differs from the launch identity")
    architectures = _gpu_architectures()
    if len(architectures) != config.tp_size:
        raise SessionWorkerError(
            "visible GPU count differs from the engine tensor-parallel degree"
        )
    facts = RuntimePreflightFacts(
        launch_digest=launch_digest,
        runtime_digest=_required_digest_env(_DIGEST_ENV["runtime_digest"]),
        stack_digest=tree.stack_digest,
        tree_digest=tree.tree_digest,
        engine_config_digest=config.digest,
        worker_distribution_digest=_required_digest_env(
            _DIGEST_ENV["worker_distribution_digest"]
        ),
        model_revision_digest=_required_digest_env(
            _DIGEST_ENV["model_revision_digest"]
        ),
        model_manifest_digest=_required_digest_env(
            _DIGEST_ENV["model_manifest_digest"]
        ),
        model_content_digest=_required_digest_env(
            _DIGEST_ENV["model_content_digest"]
        ),
        sglang_version=observed_sglang,
        gpu_architectures=architectures,
        topology_digest=_topology_digest(),
        loopback_only=True,
        read_only_inputs=True,
        private_writable_cache=True,
    )
    return facts, tree


def _prepare_descendant_bootstrap() -> None:
    """Expose only installed sitecustomize to spawned SGLang interpreters."""

    site = Path(__file__).resolve().parent / "oci_site"
    customizer = site / "sitecustomize.py"
    try:
        info = customizer.lstat()
    except OSError as exc:
        raise SessionWorkerError(f"installed OCI site bootstrap is unavailable: {exc}") from None
    if (
        stat.S_ISLNK(info.st_mode)
        or not stat.S_ISREG(info.st_mode)
        or not _path_is_read_only(customizer)
    ):
        raise SessionWorkerError("installed OCI site bootstrap has an unsafe shape")
    os.environ["PYTHONPATH"] = str(site)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ["PYTHONSAFEPATH"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import optima.bootstrap  # noqa: F401 - deliberately after live preflight


@contextlib.contextmanager
def _engine_session(config: EngineSessionConfig, tree: object) -> Iterator[object]:
    """Construct one content-selected engine without any scheduling role."""

    reference_mode = os.environ.get("OPTIMA_SESSION_PROTOCOL") == "reference"
    if reference_mode and config.seam_bindings:
        raise SessionWorkerError(
            "pristine reference engine config must not select seam bindings"
        )
    gate_environment = seam_binding_environment(
        () if reference_mode else config.seam_bindings
    )
    from optima import discovery_overlay

    discovery = _requested_discovery_overlay(reference_mode=reference_mode)
    discovery_environment = discovery_overlay.disabled_environment()
    if discovery is not None:
        overlay_root, identity = discovery
        discovery_environment = discovery_overlay.launch_environment(
            overlay_root=overlay_root,
            expected_identity_digest=identity,
        )
    try:
        with _environment(**discovery_environment):
            if discovery is not None:
                discovery_overlay.arm_driver_activation(
                    expected_identity_digest=identity,
                    expected_members=config.tp_size,
                )
                discovery_overlay.install_process_role_hook()
            with _environment(**gate_environment):
                if reference_mode:
                    os.environ["PYTHONPATH"] = ""
                else:
                    _prepare_descendant_bootstrap()
                from optima.manifest import load_manifest

                tree_root = Path(getattr(tree, "root"))
                runtime_manifest = getattr(tree, "runtime_manifest", None)
                manifest = (
                    load_manifest(tree_root)
                    if runtime_manifest is not None
                    else None
                )
            active = bool(manifest is not None and manifest.ops)
            framework_mode = bool(
                manifest is not None
                and any(operation.setup for operation in manifest.ops)
            )
            cfg = SimpleNamespace(
                model_path=config.model_path,
                dtype=config.dtype,
                deterministic=config.deterministic,
                attention_backend=config.attention_backend,
                disable_cuda_graph=config.disable_cuda_graph,
                mem_fraction_static=config.mem_fraction_static,
                log_level=config.log_level,
                max_running_requests=config.max_running_requests,
                tp_size=config.tp_size,
                moe_runner_backend=config.moe_runner_backend,
                disable_custom_all_reduce=config.disable_custom_all_reduce,
                extra_engine_kwargs=dict(config.engine_kwargs),
                seam_bindings=config.seam_bindings,
                seed=0,
                framework_mode=framework_mode,
                isolate=True,
                allow_unsafe_no_isolation=False,
            )
            from optima.eval.engine_worker import isolated_engine_session

            bundle_path = str(tree_root) if active else ""
            if manifest is not None and manifest.dep_patches and not os.environ.get(
                "FLASHINFER_WORKSPACE_BASE", ""
            ):
                raise SessionWorkerError(
                    "dep-patched tree lacks its sealed runtime workspace"
                )
            with isolated_engine_session(
                cfg,
                bundle_path=bundle_path,
                active=active,
                framework_mode=framework_mode,
                install_seams=not reference_mode,
            ) as handle:
                receipt = None
                if discovery is not None:
                    sglang_module = sys.modules.get("sglang")
                    if sglang_module is None:
                        raise SessionWorkerError(
                            "engine construction returned without stock driver SGLang"
                        )
                    receipt = discovery_overlay.require_driver_activation(
                        sglang_module,
                        overlay_root,
                        expected_identity_digest=identity,
                        expected_members=config.tp_size,
                        expected_sglang_version=_required_sglang_version(),
                    )
                yield SimpleNamespace(
                    engine=handle.engine,
                    require_completion=handle.require_completion,
                    discovery_activation_receipt=receipt,
                )
    finally:
        if discovery is not None:
            discovery_overlay.clear_driver_activation()


def _engine_outputs(outputs: object, *, request: BatchRequest) -> BatchEvidence:
    if isinstance(outputs, dict):
        rows = [outputs]
    elif isinstance(outputs, list):
        rows = outputs
    else:
        raise SessionProtocolError("engine output must be an object or array")
    if len(rows) != len(request.prompts):
        raise SessionProtocolError("engine output prompt count is invalid")
    prompts: list[PromptEvidence] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SessionProtocolError("engine output item is not an object")
        metadata = row.get("meta_info")
        if not isinstance(metadata, dict):
            raise SessionProtocolError("engine output metadata is missing")
        raw_ids = row.get("output_ids") or metadata.get("output_ids")
        raw_topk = metadata.get("output_top_logprobs")
        if not isinstance(raw_ids, (list, tuple)) or not isinstance(
            raw_topk, (list, tuple)
        ):
            raise SessionProtocolError("engine output lacks token/top-k evidence")
        output_ids: list[int] = []
        for token in raw_ids:
            if type(token) is not int:
                raise SessionProtocolError("engine output token ID is not an integer")
            output_ids.append(token)
        positions: list[tuple[tuple[float, int], ...]] = []
        for raw_position in raw_topk:
            if not isinstance(raw_position, (list, tuple)):
                raise SessionProtocolError("engine output top-k position is not an array")
            position: list[tuple[float, int]] = []
            for entry in raw_position:
                if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                    raise SessionProtocolError("engine output top-k entry is malformed")
                logprob, token_id = entry[0], entry[1]
                if (
                    isinstance(logprob, bool)
                    or not isinstance(logprob, (int, float))
                    or not math.isfinite(float(logprob))
                    or type(token_id) is not int
                ):
                    raise SessionProtocolError("engine output top-k value is invalid")
                position.append((float(logprob), token_id))
            positions.append(tuple(position))
        prompts.append(PromptEvidence(tuple(output_ids), tuple(positions)))
    return BatchEvidence(tuple(prompts))


def _generate(engine: object, request: BatchRequest) -> BatchEvidence:
    generate = getattr(engine, "generate", None)
    if not callable(generate):
        raise SessionProtocolError("engine does not expose generate()")
    outputs = generate(
        prompt=list(request.prompts),
        sampling_params={
            "temperature": request.temperature,
            "max_new_tokens": request.max_new_tokens,
            "ignore_eos": True,
        },
        return_logprob=True,
        logprob_start_len=-1,
        top_logprobs_num=request.top_logprobs_num,
    )
    return _engine_outputs(outputs, request=request)


def _canonical_prompt_ids(engine: object, prompt: str) -> list[int]:
    manager = getattr(engine, "tokenizer_manager", None)
    tokenizer = getattr(manager, "tokenizer", None)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise SessionProtocolError("pristine reference lacks the pinned tokenizer API")
    ids = encode(prompt)
    if (
        not isinstance(ids, list)
        or not ids
        or any(type(token) is not int or not 0 <= token <= 2_147_483_647 for token in ids)
    ):
        raise SessionProtocolError("pristine reference tokenization is invalid")
    return ids


def _tokenizer_vocab_size(engine: object) -> int:
    tokenizer = getattr(getattr(engine, "tokenizer_manager", None), "tokenizer", None)
    try:
        size = len(tokenizer)
    except (TypeError, AttributeError):
        size = getattr(tokenizer, "vocab_size", None)
    if type(size) is not int or not 1 <= size <= 2_147_483_648:
        raise SessionProtocolError("pristine reference vocabulary is invalid")
    return size


def _logprob_entry(value: object, *, label: str) -> tuple[float, int]:
    if not isinstance(value, (tuple, list)) or len(value) < 2:
        raise SessionProtocolError(f"pristine reference {label} entry is malformed")
    logprob, token_id = value[0], value[1]
    if (
        isinstance(logprob, bool)
        or not isinstance(logprob, (int, float))
        or not math.isfinite(float(logprob))
        or type(token_id) is not int
    ):
        raise SessionProtocolError(f"pristine reference {label} entry is invalid")
    return float(logprob), token_id


def _reference_role_evidence(
    engine: object,
    prompt_ids: list[list[int]],
    role_inputs: list[object],
    *,
    vocab_size: int,
) -> list[object]:
    from optima.eval.reference_protocol import (
        ReferenceRoleEvidence,
        ReferenceTokenEvidence,
    )

    output_ids = [list(getattr(role, "output_ids")) for role in role_inputs]
    support_ids = [
        sorted({token for row in getattr(role, "supports") for token in row})
        for role in role_inputs
    ]
    if any(
        token >= vocab_size
        for rows in (*output_ids, *support_ids)
        for token in rows
    ):
        raise SessionProtocolError("pristine reference request contains an OOV token")
    outputs = getattr(engine, "generate", None)
    if not callable(outputs):
        raise SessionProtocolError("pristine reference does not expose generate()")
    rows = outputs(
        input_ids=[
            prefix + response
            for prefix, response in zip(prompt_ids, output_ids, strict=True)
        ],
        sampling_params={"temperature": 0.0, "max_new_tokens": 0, "ignore_eos": True},
        return_logprob=True,
        logprob_start_len=[max(0, len(prefix) - 1) for prefix in prompt_ids],
        top_logprobs_num=1,
        token_ids_logprob=support_ids,
    )
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or len(rows) != len(role_inputs):
        raise SessionProtocolError("pristine reference returned wrong prompt coverage")
    result: list[object] = []
    for role, response, requested, row in zip(
        role_inputs, output_ids, support_ids, rows, strict=True
    ):
        metadata = row.get("meta_info") if isinstance(row, dict) else None
        if not isinstance(metadata, dict):
            raise SessionProtocolError("pristine reference output metadata is missing")
        targets = metadata.get("input_token_logprobs")
        top_one = metadata.get("input_top_logprobs")
        targeted = metadata.get("input_token_ids_logprobs")
        if not all(isinstance(value, list) for value in (targets, top_one, targeted)):
            raise SessionProtocolError("pristine reference omitted teacher logprobs")
        targets = targets[-len(response):]
        top_one = top_one[-len(response):]
        targeted = targeted[-len(response):]
        if not (len(targets) == len(top_one) == len(targeted) == len(response)):
            raise SessionProtocolError("pristine reference teacher coverage is incomplete")
        tokens = []
        for position, (expected, raw_target, raw_top, raw_support) in enumerate(
            zip(response, targets, top_one, targeted, strict=True)
        ):
            target_logprob, target_id = _logprob_entry(raw_target, label="target")
            if target_id != expected:
                raise SessionProtocolError("pristine reference scored the wrong target")
            if not isinstance(raw_top, list) or len(raw_top) != 1:
                raise SessionProtocolError("pristine reference true-argmax evidence is malformed")
            argmax_logprob, argmax_id = _logprob_entry(raw_top[0], label="argmax")
            if not 0 <= argmax_id < vocab_size:
                raise SessionProtocolError("pristine reference argmax is out of vocabulary")
            if not isinstance(raw_support, list):
                raise SessionProtocolError("pristine reference support evidence is malformed")
            support_map: dict[int, float] = {}
            for entry in raw_support:
                value, token_id = _logprob_entry(entry, label="support")
                if token_id in support_map:
                    raise SessionProtocolError("pristine reference duplicated a support token")
                support_map[token_id] = value
            if tuple(sorted(support_map)) != tuple(requested):
                raise SessionProtocolError("pristine reference support coverage differs")
            requested_position = getattr(role, "supports")[position]
            if expected in support_map and not math.isclose(
                support_map[expected], target_logprob, rel_tol=1e-4, abs_tol=1e-4
            ):
                raise SessionProtocolError("pristine reference target/support logprobs disagree")
            if argmax_id in support_map and not math.isclose(
                support_map[argmax_id], argmax_logprob, rel_tol=1e-4, abs_tol=1e-4
            ):
                raise SessionProtocolError("pristine reference argmax/support logprobs disagree")
            tokens.append(ReferenceTokenEvidence(
                target_logprob,
                argmax_id,
                tuple(support_map[token] for token in requested_position),
            ))
        result.append(ReferenceRoleEvidence(tuple(tokens)))
    return result


def _reference_evidence(engine: object, request: object) -> object:
    from optima.eval.reference_protocol import (
        ReferenceEvidence,
        ReferencePromptEvidence,
        request_sha256,
    )

    prompt_ids = [_canonical_prompt_ids(engine, item.prompt) for item in request.prompts]
    vocab_size = _tokenizer_vocab_size(engine)
    roles = [
        _reference_role_evidence(
            engine,
            prompt_ids,
            [prompt.roles[index] for prompt in request.prompts],
            vocab_size=vocab_size,
        )
        for index in range(3)
    ]
    prompts = []
    for index, (prompt, ids) in enumerate(zip(request.prompts, prompt_ids, strict=True)):
        token_bytes = b"".join(int(token).to_bytes(4, "big") for token in ids)
        prompts.append(ReferencePromptEvidence(
            prompt.prompt_digest,
            len(ids),
            hashlib.sha256(token_bytes).hexdigest(),
            tuple(role[index] for role in roles),
        ))
    return ReferenceEvidence(
        request.session_id,
        request.launch_digest,
        request.plan_digest,
        request_sha256(request),
        request.request_id,
        request.nonce,
        request.request_index,
        vocab_size,
        tuple(prompts),
    )


def _read_reference_request(fd: int) -> object:
    from optima.eval.reference_protocol import (
        FRAME_HEADER_BYTES as REFERENCE_HEADER_BYTES,
        MAX_REQUEST_BYTES,
        REQUEST_MAGIC,
        decode_reference_request,
    )

    header = _read_exact(fd, REFERENCE_HEADER_BYTES)
    if header[:4] != REQUEST_MAGIC:
        raise SessionProtocolError("reference request magic/version mismatch")
    size = struct.unpack(">I", header[4:8])[0]
    if size > MAX_REQUEST_BYTES:
        raise SessionProtocolError("reference request exceeds its hard bound")
    return decode_reference_request(header + _read_exact(fd, size))


def _serve_reference(
    engine: object,
    control_fd: int,
    protocol_fd: int,
    *,
    session_id: str,
    launch_digest: str,
) -> None:
    from optima.eval.reference_protocol import encode_reference_evidence

    expected_index = 0
    plan_digest: str | None = None
    seen_request_ids: set[str] = set()
    seen_nonces: set[str] = set()
    while True:
        request = _read_reference_request(control_fd)
        if plan_digest is None:
            plan_digest = request.plan_digest
        if (
            request.session_id != session_id
            or request.launch_digest != launch_digest
            or request.plan_digest != plan_digest
            or request.request_index != expected_index
            or request.request_id in seen_request_ids
            or request.nonce in seen_nonces
        ):
            raise SessionProtocolError("reference ordering, binding, or replay check failed")
        seen_request_ids.add(request.request_id)
        seen_nonces.add(request.nonce)
        evidence = _reference_evidence(engine, request)
        _write_all(protocol_fd, encode_reference_evidence(evidence, request))
        expected_index += 1


def run_session(*, input_fd: int = 0, output_fd: int | None = None) -> int:
    """Serve batches until the trusted host force-destroys the container."""

    session_protocol = os.environ.get("OPTIMA_SESSION_PROTOCOL", "ordinary")
    if session_protocol not in {"ordinary", "reference"}:
        return 1
    protocol_fd = _reserve_protocol_fd() if output_fd is None else output_fd
    os.set_inheritable(protocol_fd, False)
    control_fd = _reserve_control_fd(input_fd)
    session_id: str | None = None
    launch_digest: str | None = None
    request: BatchRequest | None = None
    stage = "init"
    try:
        init = _read_control_frame(control_fd, max_bytes=MAX_INIT_BYTES)
        session_id, launch_digest, config = validate_init(
            init,
            expected_launch_digest=os.environ.get("OPTIMA_LAUNCH_DIGEST", ""),
            expected_engine_config_digest=os.environ.get(
                "OPTIMA_ENGINE_CONFIG_DIGEST", ""
            ),
        )
        stage = "preflight"
        facts, tree = _validate_live_preflight(config, launch_digest=launch_digest)
        _write_all(
            protocol_fd,
            frame_message(
                preflight_message(
                    session_id=session_id, launch_digest=launch_digest, facts=facts
                ),
                max_bytes=MAX_CONTROL_BYTES,
            ),
        )
        stage = "preflight-accept"
        validate_preflight_accept(
            _read_control_frame(control_fd, max_bytes=MAX_CONTROL_BYTES),
            session_id=session_id,
            launch_digest=launch_digest,
            expected_facts_digest=facts.digest,
        )
        stage = "engine"
        if (
            session_protocol == "reference"
            and getattr(tree, "runtime_manifest", None) is not None
        ):
            raise SessionProtocolError(
                "pristine reference tree must contain no contribution manifest"
            )
        with _engine_session(config, tree) as handle:
            _write_all(
                protocol_fd,
                frame_message(
                    ready_message(
                        session_id=session_id,
                        launch_digest=launch_digest,
                        discovery_activation=getattr(
                            handle, "discovery_activation_receipt", None
                        ),
                    ),
                    max_bytes=MAX_CONTROL_BYTES,
                ),
            )
            if session_protocol == "reference":
                stage = "reference"
                _serve_reference(
                    handle.engine,
                    control_fd,
                    protocol_fd,
                    session_id=session_id,
                    launch_digest=launch_digest,
                )
                raise AssertionError("reference session loop returned")
            expected_index = 0
            seen_request_ids: set[str] = set()
            seen_nonces: set[str] = set()
            while True:
                stage = "batch"
                request = validate_batch_request(
                    _read_control_frame(
                        control_fd, max_bytes=MAX_BATCH_REQUEST_BYTES
                    )
                )
                if (
                    request.session_id != session_id
                    or request.launch_digest != launch_digest
                    or request.batch_index != expected_index
                    or request.request_id in seen_request_ids
                    or request.nonce in seen_nonces
                ):
                    raise SessionProtocolError(
                        "batch ordering, session, launch, or replay binding failed"
                    )
                seen_request_ids.add(request.request_id)
                seen_nonces.add(request.nonce)
                evidence = _generate(handle.engine, request)
                handle.require_completion()
                _write_all(
                    protocol_fd, evidence_frame(evidence, request=request)
                )
                expected_index += 1
                request = None
    except BaseException as exc:  # noqa: BLE001 - bounded untrusted diagnostic
        if session_id is not None and launch_digest is not None:
            try:
                _write_all(
                    protocol_fd,
                    frame_message(
                        error_message(
                            session_id=session_id,
                            launch_digest=launch_digest,
                            stage=stage,
                            error=exc,
                            request=request,
                        ),
                        max_bytes=MAX_CONTROL_BYTES,
                    ),
                )
            except BaseException:
                pass
        return 1
    finally:
        os.close(control_fd)
        if output_fd is None:
            os.close(protocol_fd)


def main() -> int:
    return run_session()


if __name__ == "__main__":  # pragma: no cover - container entry point
    raise SystemExit(main())
