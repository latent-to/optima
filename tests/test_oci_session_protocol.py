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
from optima.eval.oci_session_protocol import (
    CONTROL_MAGIC,
    EVIDENCE_MAGIC,
    FRAME_HEADER_BYTES,
    MAX_BATCH_REQUEST_BYTES,
    MAX_CONTROL_BYTES,
    MAX_ERROR_CHARS,
    MAX_INIT_BYTES,
    BatchEvidence,
    BatchRequest,
    EngineSessionConfig,
    PromptEvidence,
    RuntimePreflightFacts,
    SessionProtocolError,
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
    validate_ready(message, session_id=SESSION, launch_digest=LAUNCH)
    with pytest.raises(SessionProtocolError, match="stale"):
        validate_ready(message, session_id="4" * 32, launch_digest=LAUNCH)
    changed = dict(message, engine_seconds=1.0)
    with pytest.raises(SessionProtocolError, match="stale"):
        validate_ready(changed, session_id=SESSION, launch_digest=LAUNCH)


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
        ({"launch_digest": "0" * 64}, "nonzero"),
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
