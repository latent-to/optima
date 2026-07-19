from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import struct
from dataclasses import replace
from pathlib import Path

import pytest

import optima.eval.oci_session_protocol as protocol
from optima.discovery_overlay import (
    DiscoveryActivationReceipt,
    DiscoveryDriverOrigin,
    DiscoverySchedulerMember,
)
from optima.seams import (
    SEAM_ADAPTERS,
    SEAM_BINDINGS,
    SEAM_BINDING_ENV_GATES,
    normalize_seam_bindings,
    seam_binding_environment,
)
from optima.eval.oci_session_protocol import (
    CONTROL_MAGIC,
    EVIDENCE_MAGIC,
    FRAME_HEADER_BYTES,
    MAX_BATCH_REQUEST_BYTES,
    MAX_CONTROL_BYTES,
    MAX_ERROR_CHARS,
    MAX_INIT_BYTES,
    AuditReceiptFacts,
    BatchEvidence,
    BatchRequest,
    EngineSessionConfig,
    PromptEvidence,
    RuntimePreflightFacts,
    SessionProtocolError,
    SlotAuditControl,
    SlotAuditPolicy,
    audit_evidence_message,
    audit_policy_from_init,
    batch_request,
    decode_message,
    encode_message,
    error_message,
    evidence_frame,
    expected_evidence_payload_bytes,
    frame_message,
    make_init,
    parse_error_message,
    parse_evidence_frame_bytes,
    parse_frame_bytes,
    preflight_accept_message,
    preflight_message,
    ready_message,
    validate_batch_request,
    validate_audit_evidence,
    validate_init,
    validate_preflight,
    validate_preflight_accept,
    validate_ready,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


SESSION = "1" * 32
REQUEST = "2" * 32
NONCE = "3" * 32
LAUNCH = _digest("launch")


def _discovery_receipt(**changes: object) -> DiscoveryActivationReceipt:
    values: dict[str, object] = {
        "schema": "optima.discovery-driver-activation.v1",
        "overlay_identity_digest": _digest("discovery-overlay"),
        "driver_pid": 100,
        "driver_origin": DiscoveryDriverOrigin(
            "sglang", "0.0.0.dev1+g56e290315", "sglang/__init__.py"
        ),
        "scheduler_target_module": "sglang.srt.managers.scheduler",
        "scheduler_target_qualname": "run_scheduler_process",
        "tp_size": 2,
        "members": (
            DiscoverySchedulerMember(201, 0, 0, 0, 0, 0, 0, None),
            DiscoverySchedulerMember(202, 1, 1, 1, 0, 1, 0, None),
        ),
    }
    values.update(changes)
    return DiscoveryActivationReceipt(**values)  # type: ignore[arg-type]


def _config(**changes: object) -> EngineSessionConfig:
    values: dict[str, object] = {
        "model_path": protocol.CONTAINER_MODEL_PATH,
        "dtype": "bfloat16",
        "deterministic": False,
        "attention_backend": "flashinfer",
        "disable_cuda_graph": False,
        "mem_fraction_static": 0.82,
        "log_level": "error",
        "max_running_requests": 64,
        "tp_size": 8,
        "moe_runner_backend": "flashinfer_trtllm",
        "disable_custom_all_reduce": False,
        "engine_kwargs": {
            "page_size": 64,
            "enable_flashinfer_allreduce_fusion": True,
        },
        "seam_bindings": (),
    }
    values.update(changes)
    return EngineSessionConfig(**values)  # type: ignore[arg-type]


def _facts(**changes: object) -> RuntimePreflightFacts:
    values: dict[str, object] = {
        "launch_digest": LAUNCH,
        "runtime_digest": _digest("runtime"),
        "stack_digest": _digest("stack"),
        "tree_digest": _digest("tree"),
        "engine_config_digest": _config().digest,
        "worker_distribution_digest": _digest("worker"),
        "model_revision_digest": _digest("revision"),
        "model_manifest_digest": _digest("manifest"),
        "model_content_digest": _digest("content"),
        "sglang_version": "0.0.0.dev1+g56e290315",
        "gpu_architectures": ("sm120",) * 8,
        "topology_digest": _digest("topology"),
        "loopback_only": True,
        "read_only_inputs": True,
        "private_writable_cache": True,
    }
    values.update(changes)
    return RuntimePreflightFacts(**values)  # type: ignore[arg-type]


def _request(**changes: object) -> BatchRequest:
    values: dict[str, object] = {
        "session_id": SESSION,
        "launch_digest": LAUNCH,
        "request_id": REQUEST,
        "nonce": NONCE,
        "batch_index": 0,
        "prompts": ("alpha", "beta"),
        "max_new_tokens": 2,
        "top_logprobs_num": 2,
        "temperature": 0.0,
    }
    values.update(changes)
    return BatchRequest(**values)  # type: ignore[arg-type]


def _evidence() -> BatchEvidence:
    positions = (
        ((-0.4, 10), (-1.2, 11)),
        ((-0.5, 20), (-1.1, 21)),
    )
    return BatchEvidence(
        (
            PromptEvidence((10, 20), positions),
            PromptEvidence((30, 40), positions),
        )
    )


def test_engine_config_is_exact_immutable_and_digest_stable() -> None:
    source = {
        "page_size": 64,
        "enable_flashinfer_allreduce_fusion": True,
    }
    config = _config(engine_kwargs=source)
    source["page_size"] = 128

    assert EngineSessionConfig.from_dict(config.to_dict()) == config
    assert config.engine_kwargs["page_size"] == 64
    with pytest.raises(TypeError):
        config.engine_kwargs["page_size"] = 128  # type: ignore[index]
    assert config.digest == EngineSessionConfig.from_dict(config.to_dict()).digest
    assert len(config.digest) == 64


def test_seam_binding_table_is_closed_and_deep_epilogue_shares_arfusion() -> None:
    assert dict(SEAM_BINDING_ENV_GATES) == {
        "arfusion": "OPTIMA_ARFUSION_SEAM",
        "attention": "OPTIMA_ATTENTION_SEAM",
        "collective": "OPTIMA_COLLECTIVE_SEAM",
        "moe": "OPTIMA_MOE_SEAM",
        "msa_prefill": "OPTIMA_MSA_PREFILL_SEAM",
    }
    bindings = {binding.binding_id: binding for binding in SEAM_BINDINGS}
    assert bindings["arfusion"].adapters == (
        "arfusion",
        "defer_gate",
        "moe_export",
    )
    for binding in SEAM_BINDINGS:
        adapter_rows = tuple(
            adapter
            for adapter in SEAM_ADAPTERS
            if adapter.binding_id == binding.binding_id
        )
        assert tuple(adapter.name for adapter in adapter_rows) == binding.adapters
        assert {adapter.environment_gate for adapter in adapter_rows} == {
            binding.environment_gate
        }


def test_seam_bindings_normalize_and_emit_complete_explicit_environment() -> None:
    selected = normalize_seam_bindings(["arfusion", "msa_prefill"])
    assert selected == ("arfusion", "msa_prefill")
    assert seam_binding_environment(selected) == {
        "OPTIMA_ARFUSION_SEAM": "1",
        "OPTIMA_ATTENTION_SEAM": "0",
        "OPTIMA_COLLECTIVE_SEAM": "0",
        "OPTIMA_MOE_SEAM": "0",
        "OPTIMA_MSA_PREFILL_SEAM": "1",
    }
    assert seam_binding_environment(()) == {
        "OPTIMA_ARFUSION_SEAM": "0",
        "OPTIMA_ATTENTION_SEAM": "0",
        "OPTIMA_COLLECTIVE_SEAM": "0",
        "OPTIMA_MOE_SEAM": "0",
        "OPTIMA_MSA_PREFILL_SEAM": "0",
    }


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("attention", "array"),
        ({"attention"}, "array"),
        (("unknown",), "unknown"),
        (("moe", "attention"), "sorted"),
        (("attention", "attention"), "duplicates"),
        (("attention", 1), "strings"),
    ],
)
def test_seam_bindings_reject_noncanonical_or_open_input(
    value: object, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        normalize_seam_bindings(value)
    with pytest.raises(ValueError, match=match):
        seam_binding_environment(value)


def test_engine_config_binds_seams_in_wire_and_digest() -> None:
    config = _config(seam_bindings=("attention", "collective"))
    row = config.to_dict()
    assert row["seam_bindings"] == ["attention", "collective"]
    assert EngineSessionConfig.from_dict(row) == config
    assert config.digest != _config().digest

    row["seam_bindings"] = ("attention",)
    with pytest.raises(SessionProtocolError, match="array"):
        EngineSessionConfig.from_dict(row)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"model_path": "/host/model"}, "model_path"),
        ({"dtype": "auto"}, "dtype"),
        ({"deterministic": 1}, "deterministic"),
        ({"attention_backend": "bad value"}, "attention_backend"),
        ({"disable_cuda_graph": 0}, "disable_cuda_graph"),
        ({"mem_fraction_static": float("nan")}, "finite"),
        ({"mem_fraction_static": 1.0}, "finite"),
        ({"max_running_requests": 0}, "max_running_requests"),
        ({"tp_size": True}, "tp_size"),
        ({"moe_runner_backend": "x\n"}, "moe_runner_backend"),
        ({"engine_kwargs": {"arbitrary": True}}, "unsupported keys"),
        ({"engine_kwargs": {"page_size": False}}, "page_size"),
        ({"seam_bindings": "attention"}, "array"),
        ({"seam_bindings": ("moe", "attention")}, "sorted"),
        ({"seam_bindings": ("attention", "attention")}, "duplicates"),
        ({"seam_bindings": ("unknown",)}, "unknown"),
    ],
)
def test_engine_config_rejects_invalid_and_unreviewed_fields(
    changes: dict[str, object], match: str
) -> None:
    with pytest.raises(SessionProtocolError, match=match):
        _config(**changes)

    row = _config().to_dict()
    row["prompt_seed"] = 7
    with pytest.raises(SessionProtocolError, match="fields"):
        EngineSessionConfig.from_dict(row)


def test_init_is_bound_to_launch_and_exact_engine_config() -> None:
    config = _config()
    message = make_init(
        config,
        session_id=SESSION,
        launch_digest=LAUNCH,
        expected_engine_config_digest=config.digest,
    )
    session, launch, decoded = validate_init(
        message,
        expected_launch_digest=LAUNCH,
        expected_engine_config_digest=config.digest,
    )
    assert (session, launch, decoded) == (SESSION, LAUNCH, config)

    wire = json.dumps(message, sort_keys=True)
    for forbidden in (
        "prompt_seed",
        "prompt_plan",
        "entropy",
        "threshold",
        "hidden_quality",
        '"score"',
        '"mode"',
        '"role"',
    ):
        assert forbidden not in wire

    changed = copy.deepcopy(message)
    changed["engine_config"]["tp_size"] = 4
    with pytest.raises(SessionProtocolError, match="digest mismatch"):
        validate_init(changed)
    changed = copy.deepcopy(message)
    changed["mode"] = "candidate"
    with pytest.raises(SessionProtocolError, match="fields"):
        validate_init(changed)
    with pytest.raises(SessionProtocolError, match="launch binding"):
        validate_init(message, expected_launch_digest=_digest("other"))


def test_strict_control_json_duplicate_nonfinite_nesting_and_trailing() -> None:
    assert decode_message(b'{"a":1}', max_bytes=32) == {"a": 1}
    with pytest.raises(SessionProtocolError, match="duplicate JSON key"):
        decode_message(b'{"a":1,"a":2}', max_bytes=32)
    with pytest.raises(SessionProtocolError, match="non-finite"):
        decode_message(b'{"a":NaN}', max_bytes=32)
    with pytest.raises(SessionProtocolError, match="trailing"):
        decode_message(b'{"a":1} ', max_bytes=32)
    with pytest.raises(SessionProtocolError, match="nesting"):
        decode_message(
            (b'{"a":' + b"[" * 25 + b"0" + b"]" * 25 + b"}"),
            max_bytes=256,
        )
    with pytest.raises(SessionProtocolError, match="integer"):
        decode_message(b'{"a":18446744073709551616}', max_bytes=64)
    with pytest.raises(SessionProtocolError, match="non-finite"):
        encode_message({"x": float("inf")}, max_bytes=64)


def test_control_frame_rejects_partial_wrong_magic_oversize_and_trailing() -> None:
    message = ready_message(session_id=SESSION, launch_digest=LAUNCH)
    frame = frame_message(message, max_bytes=MAX_CONTROL_BYTES)
    assert parse_frame_bytes(frame, max_bytes=MAX_CONTROL_BYTES) == message

    for broken, match in (
        (frame[:7], "truncated"),
        (b"NOPE" + frame[4:], "magic"),
        (frame + b"x", "trailing"),
        (frame[:-1], "missing"),
    ):
        with pytest.raises(SessionProtocolError, match=match):
            parse_frame_bytes(broken, max_bytes=MAX_CONTROL_BYTES)
    declared = CONTROL_MAGIC + struct.pack(">I", MAX_CONTROL_BYTES + 1)
    with pytest.raises(SessionProtocolError, match="oversized"):
        parse_frame_bytes(declared, max_bytes=MAX_CONTROL_BYTES)


def test_preflight_is_exact_raw_and_host_compared() -> None:
    facts = _facts()
    message = preflight_message(
        session_id=SESSION, launch_digest=LAUNCH, facts=facts
    )
    assert validate_preflight(
        message,
        session_id=SESSION,
        launch_digest=LAUNCH,
        expected_facts=facts,
    ) == facts
    assert "verified" not in message["facts"]
    assert "passed" not in message["facts"]

    changed = copy.deepcopy(message)
    changed["facts"]["tree_digest"] = _digest("wrong-tree")
    with pytest.raises(SessionProtocolError, match="host policy"):
        validate_preflight(
            changed,
            session_id=SESSION,
            launch_digest=LAUNCH,
            expected_facts=facts,
        )
    with pytest.raises(SessionProtocolError, match="not proven"):
        _facts(loopback_only=False)
    with pytest.raises(SessionProtocolError, match="gpu_architectures"):
        _facts(gpu_architectures=("sm120", "bad"))


def test_preflight_accept_is_bound_to_exact_accepted_facts() -> None:
    facts = _facts()
    message = preflight_accept_message(
        session_id=SESSION, launch_digest=LAUNCH, facts=facts
    )
    assert message["facts_digest"] == facts.digest
    validate_preflight_accept(
        message,
        session_id=SESSION,
        launch_digest=LAUNCH,
        expected_facts_digest=facts.digest,
    )
    for changed in (
        dict(message, session_id="4" * 32),
        dict(message, launch_digest=_digest("other-launch")),
        dict(message, facts_digest=_digest("other-facts")),
        dict(message, accepted=True),
    ):
        with pytest.raises(SessionProtocolError, match="stale|malformed"):
            validate_preflight_accept(
                changed,
                session_id=SESSION,
                launch_digest=LAUNCH,
                expected_facts_digest=facts.digest,
            )


def test_ready_is_only_an_exact_bound_marker() -> None:
    message = ready_message(session_id=SESSION, launch_digest=LAUNCH)
    expected_payload = (
        b'{"launch_digest":"' + LAUNCH.encode("ascii")
        + b'","schema":"optima-isolated-engine-session-v1","session_id":"'
        + SESSION.encode("ascii") + b'","type":"ready"}'
    )
    assert encode_message(message, max_bytes=MAX_CONTROL_BYTES) == expected_payload
    assert frame_message(message, max_bytes=MAX_CONTROL_BYTES) == (
        CONTROL_MAGIC + struct.pack(">I", len(expected_payload)) + expected_payload
    )
    assert validate_ready(
        message, session_id=SESSION, launch_digest=LAUNCH
    ) is None
    with pytest.raises(SessionProtocolError, match="stale"):
        validate_ready(message, session_id="4" * 32, launch_digest=LAUNCH)
    changed = dict(message, engine_seconds=1.0)
    with pytest.raises(SessionProtocolError, match="stale"):
        validate_ready(changed, session_id=SESSION, launch_digest=LAUNCH)


def test_discovery_ready_is_strictly_bound_and_never_accepted_by_ordinary() -> None:
    receipt = _discovery_receipt()
    message = ready_message(
        session_id=SESSION,
        launch_digest=LAUNCH,
        discovery_activation=receipt,
    )
    assert validate_ready(
        message,
        session_id=SESSION,
        launch_digest=LAUNCH,
        expected_discovery_identity_digest=receipt.overlay_identity_digest,
        expected_discovery_tp_size=receipt.tp_size,
        expected_discovery_sglang_version=receipt.driver_origin.version,
    ) == receipt
    with pytest.raises(SessionProtocolError, match="stale|malformed"):
        validate_ready(message, session_id=SESSION, launch_digest=LAUNCH)
    with pytest.raises(SessionProtocolError, match="fields"):
        validate_ready(
            ready_message(session_id=SESSION, launch_digest=LAUNCH),
            session_id=SESSION,
            launch_digest=LAUNCH,
            expected_discovery_identity_digest=receipt.overlay_identity_digest,
            expected_discovery_tp_size=receipt.tp_size,
            expected_discovery_sglang_version=receipt.driver_origin.version,
        )

    for changes in (
        {"expected_discovery_identity_digest": _digest("other-overlay")},
        {"expected_discovery_tp_size": 3},
        {"expected_discovery_sglang_version": "9.9.9"},
    ):
        expected = {
            "expected_discovery_identity_digest": receipt.overlay_identity_digest,
            "expected_discovery_tp_size": receipt.tp_size,
            "expected_discovery_sglang_version": receipt.driver_origin.version,
            **changes,
        }
        with pytest.raises(SessionProtocolError, match="host policy"):
            validate_ready(
                message,
                session_id=SESSION,
                launch_digest=LAUNCH,
                **expected,
            )


def test_discovery_ready_rejects_partial_expectation_and_extra_receipt_fields() -> None:
    receipt = _discovery_receipt()
    message = ready_message(
        session_id=SESSION,
        launch_digest=LAUNCH,
        discovery_activation=receipt,
    )
    with pytest.raises(SessionProtocolError, match="expectation is incomplete"):
        validate_ready(
            message,
            session_id=SESSION,
            launch_digest=LAUNCH,
            expected_discovery_identity_digest=receipt.overlay_identity_digest,
        )
    changed = copy.deepcopy(message)
    changed["discovery_activation"]["candidate_claim"] = True
    with pytest.raises(SessionProtocolError, match="malformed"):
        validate_ready(
            changed,
            session_id=SESSION,
            launch_digest=LAUNCH,
            expected_discovery_identity_digest=receipt.overlay_identity_digest,
            expected_discovery_tp_size=receipt.tp_size,
            expected_discovery_sglang_version=receipt.driver_origin.version,
        )


def test_batch_request_discloses_only_one_exact_bounded_batch() -> None:
    message = batch_request(
        session_id=SESSION,
        launch_digest=LAUNCH,
        request_id=REQUEST,
        nonce=NONCE,
        batch_index=0,
        prompts=("alpha", "beta"),
        max_new_tokens=2,
        top_logprobs_num=2,
        temperature=0.0,
    )
    request = validate_batch_request(message)
    assert request == _request()
    assert "future_prompts" not in message
    assert "warmup" not in message
    assert "timed" not in message
    assert "seed" not in message

    changed = dict(message, prompts="alpha")
    with pytest.raises(SessionProtocolError, match="array"):
        validate_batch_request(changed)
    changed = dict(message, prompt_plan=[])
    with pytest.raises(SessionProtocolError, match="fields"):
        validate_batch_request(changed)
    changed = dict(message, temperature=float("nan"))
    with pytest.raises(SessionProtocolError, match="finite"):
        validate_batch_request(changed)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"session_id": "0" * 32}, "nonzero"),
        ({"request_id": SESSION}, "distinct"),
        ({"nonce": REQUEST}, "distinct"),
        ({"launch_digest": "0" * 64}, "all-zero"),
        ({"batch_index": -1}, "batch_index"),
        ({"prompts": ()}, "prompts count"),
        ({"prompts": (3,)}, "invalid"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"top_logprobs_num": 0}, "top_logprobs_num"),
        ({"temperature": float("inf")}, "finite"),
    ],
)
def test_batch_request_binding_and_shape_fail_closed(
    changes: dict[str, object], match: str
) -> None:
    with pytest.raises(SessionProtocolError, match=match):
        _request(**changes)


def test_binary_evidence_is_fixed_size_token_topk_only() -> None:
    request = _request()
    evidence = _evidence()
    frame = evidence_frame(evidence, request=request)

    assert frame[:4] == EVIDENCE_MAGIC
    assert len(frame) == FRAME_HEADER_BYTES + expected_evidence_payload_bytes(request)
    assert struct.unpack(">I", frame[4:8])[0] == expected_evidence_payload_bytes(
        request
    )
    decoded = parse_evidence_frame_bytes(frame, request=request)
    assert decoded.observed_tokens == 4
    assert tuple(prompt.output_ids for prompt in decoded.prompts) == (
        (10, 20),
        (30, 40),
    )
    assert decoded.prompts[0].top_logprobs[0][0][0] == pytest.approx(-0.4)

    # No string payload can exist after the fixed binding: every remaining byte
    # is accounted for by integer token IDs and float32/token-ID top-k pairs.
    per_position = 4 + request.top_logprobs_num * 8
    assert expected_evidence_payload_bytes(request) == (
        protocol._EVIDENCE_BINDING.size
        + len(request.prompts) * request.max_new_tokens * per_position
    )


def test_binary_evidence_rejects_binding_shape_magic_and_trailing_corruption() -> None:
    request = _request()
    frame = evidence_frame(_evidence(), request=request)
    wrong_requests = (
        replace(request, session_id="4" * 32),
        replace(request, launch_digest=_digest("wrong")),
        replace(request, request_id="4" * 32),
        replace(request, nonce="4" * 32),
        replace(request, batch_index=1),
    )
    for wrong in wrong_requests:
        with pytest.raises(SessionProtocolError, match="binding"):
            parse_evidence_frame_bytes(frame, request=wrong)

    for broken, match in (
        (frame[:7], "truncated"),
        (b"NOPE" + frame[4:], "magic"),
        (frame + b"x", "trailing"),
        (frame[:-1], "missing"),
    ):
        with pytest.raises(SessionProtocolError, match=match):
            parse_evidence_frame_bytes(broken, request=request)
    wrong_size = EVIDENCE_MAGIC + struct.pack(">I", 1) + frame[8:]
    with pytest.raises(SessionProtocolError, match="wrong exact size"):
        parse_evidence_frame_bytes(wrong_size, request=request)


@pytest.mark.parametrize(
    ("evidence", "match"),
    [
        (
            BatchEvidence((PromptEvidence((1,), (((-0.2, 1), (-1.0, 2)),)),) * 2),
            "short/oversized",
        ),
        (
            BatchEvidence(
                (
                    PromptEvidence(
                        (1, 2),
                        (((float("nan"), 1), (-1.0, 2)),) * 2,
                    ),
                )
                * 2
            ),
            "finite",
        ),
        (
            BatchEvidence(
                (
                    PromptEvidence(
                        (1, 2),
                        (((-0.2, 1), (-1.0, 1)),) * 2,
                    ),
                )
                * 2
            ),
            "duplicated",
        ),
        (
            BatchEvidence(
                (
                    PromptEvidence(
                        (1, 2),
                        (((-1.0, 1), (-0.2, 2)),) * 2,
                    ),
                )
                * 2
            ),
            "descending",
        ),
        (
            BatchEvidence(
                (
                    PromptEvidence(
                        (1, 2),
                        (((-0.01, 1), (-0.01, 2)),) * 2,
                    ),
                )
                * 2
            ),
            "mass",
        ),
    ],
)
def test_binary_evidence_values_fail_closed(
    evidence: BatchEvidence, match: str
) -> None:
    with pytest.raises(SessionProtocolError, match=match):
        evidence_frame(evidence, request=_request())


def test_binary_decoder_rejects_nonfinite_float_even_at_exact_size() -> None:
    request = _request()
    frame = bytearray(evidence_frame(_evidence(), request=request))
    first_logprob = (
        FRAME_HEADER_BYTES + protocol._EVIDENCE_BINDING.size + protocol._TOKEN_ID.size
    )
    frame[first_logprob : first_logprob + 4] = struct.pack(">f", math.nan)
    with pytest.raises(SessionProtocolError, match="finite"):
        parse_evidence_frame_bytes(bytes(frame), request=request)


def test_error_frame_is_bounded_and_exactly_bound_without_authority() -> None:
    request = _request()
    message = error_message(
        session_id=SESSION,
        launch_digest=LAUNCH,
        stage="batch-0",
        error=RuntimeError("x" * (MAX_ERROR_CHARS + 20)),
        request=request,
    )
    assert parse_error_message(
        message,
        session_id=SESSION,
        launch_digest=LAUNCH,
        request=request,
    ) == ("batch-0", "RuntimeError", "x" * MAX_ERROR_CHARS)
    assert not ({"passed", "score", "crown", "worker_seconds"} & set(message))

    assert parse_error_message(
        ready_message(session_id=SESSION, launch_digest=LAUNCH),
        session_id=SESSION,
        launch_digest=LAUNCH,
    ) is None
    with pytest.raises(SessionProtocolError, match="stale"):
        parse_error_message(
            message,
            session_id=SESSION,
            launch_digest=LAUNCH,
            request=replace(request, nonce="4" * 32),
        )
    changed = dict(message, traceback="forbidden")
    with pytest.raises(SessionProtocolError, match="fields"):
        parse_error_message(
            changed,
            session_id=SESSION,
            launch_digest=LAUNCH,
            request=request,
        )

    class HostileError(RuntimeError):
        def __str__(self) -> str:
            raise AssertionError("candidate exception rendering executed")

    hostile = error_message(
        session_id=SESSION,
        launch_digest=LAUNCH,
        stage="batch-0",
        error=HostileError(object()),
        request=request,
    )
    assert hostile["error_type"] == "HostileError"
    assert hostile["message"] == "<non-primitive exception detail omitted>"


def test_protocol_module_has_no_evaluator_runtime_quality_or_chain_import() -> None:
    path = Path(protocol.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    forbidden = {
        "optima.eval.oci_protocol",
        "optima.eval.throughput_kl",
        "optima.eval.scoring",
        "optima.eval.kl",
        "optima.eval.external_quality",
        "optima.chain",
        "torch",
        "sglang",
    }
    assert not (imports & forbidden)


def test_protocol_control_limits_are_separate_and_bounded() -> None:
    assert MAX_INIT_BYTES <= MAX_CONTROL_BYTES
    assert MAX_CONTROL_BYTES < MAX_BATCH_REQUEST_BYTES
    assert len(frame_message(
        ready_message(session_id=SESSION, launch_digest=LAUNCH),
        max_bytes=MAX_CONTROL_BYTES,
    )) > FRAME_HEADER_BYTES


def test_slot_audit_policy_init_and_raw_evidence_are_exactly_bound() -> None:
    config = _config()
    policy = SlotAuditPolicy(
        "a" * 32, 125_000, 32, ("norm.rmsnorm",), config.tp_size
    )
    assert policy.control == SlotAuditControl(
        125_000, 32, ("norm.rmsnorm",), config.tp_size
    )
    assert SlotAuditControl.from_dict(policy.control.to_dict()) == policy.control
    assert replace(policy, validator_seed="b" * 32).control.digest == policy.control.digest
    init = make_init(
        config,
        session_id=SESSION,
        launch_digest=LAUNCH,
        expected_engine_config_digest=config.digest,
        audit_policy=policy,
    )
    assert audit_policy_from_init(init) == policy
    assert validate_init(
        init, expected_audit_policy_digest=policy.digest
    ) == (SESSION, LAUNCH, config)

    request = validate_batch_request(
        batch_request(
            session_id=SESSION,
            launch_digest=LAUNCH,
            request_id=REQUEST,
            nonce=NONCE,
            batch_index=0,
            prompts=("prompt",),
            max_new_tokens=2,
            top_logprobs_num=1,
            temperature=0.0,
        )
    )
    receipts = tuple(
        AuditReceiptFacts(
            "norm.rmsnorm", 40, 0, 0, 0, 1.0, 0.995,
            "allclose", 100 + rank, rank, config.tp_size,
        )
        for rank in range(config.tp_size)
    )
    message = audit_evidence_message(
        request=request, policy=policy, receipts=receipts
    )
    assert validate_audit_evidence(
        message, request=request, policy=policy
    ) == receipts

    with pytest.raises(SessionProtocolError, match="binding"):
        validate_audit_evidence(
            {**message, "nonce": "4" * 32}, request=request, policy=policy
        )
    with pytest.raises(SessionProtocolError, match="binding"):
        validate_audit_evidence(
            message,
            request=request,
            policy=replace(policy, validator_seed="b" * 32),
        )


def test_slot_audit_policy_and_receipts_reject_ambiguous_coverage() -> None:
    with pytest.raises(SessionProtocolError, match="sorted unique"):
        SlotAuditPolicy(
            "a" * 32, 1, 1, ("norm.rmsnorm", "norm.rmsnorm"), 1
        )
    with pytest.raises(SessionProtocolError, match="rank"):
        AuditReceiptFacts(
            "norm.rmsnorm", 1, 0, 0, 0, 1.0, 0.995,
            "allclose", 100, 1, 1,
        )
    with pytest.raises(SessionProtocolError, match="internally inconsistent"):
        AuditReceiptFacts(
            "norm.rmsnorm", 32, 0, 0, 0, 0.1, 0.995,
            "allclose", 100, 0, 1,
        )
