"""Exact binary wire for one candidate-free pristine-reference session.

The controller derives requests from retained B/C/B-prime trajectories.  The
worker returns only primitive teacher logprobs; it never receives a verdict,
quality threshold, hidden judge, or miner identity.  Session orchestration and
the final ``t_session_digest`` deliberately live outside this module.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from optima.stack_identity import sha256_hex
from optima._strict import require_digest, require_int


REQUEST_MAGIC = b"ORQ1"
EVIDENCE_MAGIC = b"ORE1"
FRAME_HEADER_BYTES = 8
MAX_REQUEST_BYTES = 128 * 1024 * 1024
MAX_EVIDENCE_BYTES = 512 * 1024 * 1024
MAX_PROMPTS = 4096
MAX_PROMPT_BYTES = 2_000_000
MAX_TOTAL_PROMPT_BYTES = 96_000_000
MAX_TOKENS = 32_768
MAX_SUPPORT_WIDTH = 256
MAX_SUPPORT_UNION = 4096
MAX_DERIVED_LOGPROBS = 16_777_216
MAX_TOKEN_ID = 2_147_483_647
MAX_INDEX = 0xFFFFFFFE

ROLE_NAMES = ("baseline", "candidate", "stock_control")
_ROLE_COUNT = len(ROLE_NAMES)
_LOGPROB_MIN = -1_000_000.0
_LOGPROB_MAX = 1e-4

_REQUEST_HEADER = struct.Struct(">16s32s32s16s16sIIIH2x")
_PROMPT_HEADER = struct.Struct(">32sI")
_U32 = struct.Struct(">I")
_EVIDENCE_HEADER = struct.Struct(">16s32s32s32s16s16sIIIIH2x")
_EVIDENCE_PROMPT = struct.Struct(">32sI32s")
_TOKEN_EVIDENCE = struct.Struct(">fI")
_F32 = struct.Struct(">f")


class ReferenceProtocolError(ValueError):
    """A reference frame is malformed, ambiguous, or outside its bounds."""


def _digest(value: object, field: str) -> str:
    return require_digest(value, field=field, error=ReferenceProtocolError)


def _binding(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 32
        or any(char not in "0123456789abcdef" for char in value)
        or value == "0" * 32
    ):
        raise ReferenceProtocolError(f"{field} must be nonzero lowercase 128-bit hex")
    return value


def _integer(value: object, field: str, low: int, high: int) -> int:
    return require_int(value, field=field, error=ReferenceProtocolError, minimum=low, maximum=high)


def _tuple(value: object, field: str) -> tuple:
    if not isinstance(value, (tuple, list)):
        raise ReferenceProtocolError(f"{field} must be an array")
    return tuple(value)


def _f32(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceProtocolError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not _LOGPROB_MIN <= result <= _LOGPROB_MAX:
        raise ReferenceProtocolError(f"{field} is nonfinite or outside the logprob bound")
    return _F32.unpack(_F32.pack(result))[0]


def _roles(value: object, expected: type, field: str) -> tuple:
    rows = _tuple(value, field)
    if len(rows) != _ROLE_COUNT or any(type(row) is not expected for row in rows):
        raise ReferenceProtocolError(f"{field} must contain baseline/candidate/stock-control")
    return rows


@dataclass(frozen=True)
class ReferenceRoleInput:
    output_ids: tuple[int, ...]
    supports: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        outputs = tuple(
            _integer(value, "output token ID", 0, MAX_TOKEN_ID)
            for value in _tuple(self.output_ids, "output IDs")
        )
        supports = tuple(
            tuple(
                _integer(value, "support token ID", 0, MAX_TOKEN_ID)
                for value in _tuple(row, "support row")
            )
            for row in _tuple(self.supports, "supports")
        )
        if not outputs or len(supports) != len(outputs):
            raise ReferenceProtocolError("role output/support token coverage differs")
        if any(not row or row != tuple(sorted(set(row))) for row in supports):
            raise ReferenceProtocolError("support rows must be nonempty, unique, and token-sorted")
        object.__setattr__(self, "output_ids", outputs)
        object.__setattr__(self, "supports", supports)


@dataclass(frozen=True)
class ReferencePromptInput:
    prompt_digest: str
    prompt: str
    roles: tuple[ReferenceRoleInput, ReferenceRoleInput, ReferenceRoleInput]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_digest", _digest(self.prompt_digest, "prompt digest"))
        if not isinstance(self.prompt, str) or not self.prompt or "\x00" in self.prompt:
            raise ReferenceProtocolError("reference prompt must be nonempty text without NUL")
        try:
            encoded = self.prompt.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ReferenceProtocolError(f"reference prompt is not UTF-8: {exc}") from None
        if len(encoded) > MAX_PROMPT_BYTES:
            raise ReferenceProtocolError("reference prompt exceeds its byte bound")
        object.__setattr__(self, "roles", _roles(self.roles, ReferenceRoleInput, "prompt roles"))


@dataclass(frozen=True)
class ReferenceRequest:
    session_id: str
    launch_digest: str
    plan_digest: str
    request_id: str
    nonce: str
    request_index: int
    tokens_per_prompt: int
    support_width: int
    prompts: tuple[ReferencePromptInput, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _binding(self.session_id, "session ID"))
        object.__setattr__(self, "launch_digest", _digest(self.launch_digest, "launch digest"))
        object.__setattr__(self, "plan_digest", _digest(self.plan_digest, "plan digest"))
        object.__setattr__(self, "request_id", _binding(self.request_id, "request ID"))
        object.__setattr__(self, "nonce", _binding(self.nonce, "nonce"))
        if len({self.session_id, self.request_id, self.nonce}) != 3:
            raise ReferenceProtocolError("session, request, and nonce bindings must be distinct")
        object.__setattr__(self, "request_index", _integer(self.request_index, "request index", 0, MAX_INDEX))
        object.__setattr__(self, "tokens_per_prompt", _integer(self.tokens_per_prompt, "token count", 1, MAX_TOKENS))
        object.__setattr__(self, "support_width", _integer(self.support_width, "support width", 1, MAX_SUPPORT_WIDTH))
        prompts = _tuple(self.prompts, "reference prompts")
        if not 1 <= len(prompts) <= MAX_PROMPTS or any(type(row) is not ReferencePromptInput for row in prompts):
            raise ReferenceProtocolError("reference prompt coverage is invalid")
        keys = tuple(row.prompt_digest for row in prompts)
        if keys != tuple(sorted(set(keys))):
            raise ReferenceProtocolError("reference prompts must be unique and digest-sorted")
        total_text = sum(len(row.prompt.encode("utf-8")) for row in prompts)
        if total_text > MAX_TOTAL_PROMPT_BYTES:
            raise ReferenceProtocolError("reference prompt bytes exceed the aggregate bound")
        for prompt in prompts:
            for role in prompt.roles:
                if len(role.output_ids) != self.tokens_per_prompt or any(
                    len(row) != self.support_width for row in role.supports
                ):
                    raise ReferenceProtocolError("request role geometry differs from its header")
        derived = 0
        for prompt in prompts:
            for role in prompt.roles:
                support_union = {token for row in role.supports for token in row}
                if len(support_union) > MAX_SUPPORT_UNION:
                    raise ReferenceProtocolError("reference support union exceeds its hard bound")
                derived += len(role.output_ids) * len(support_union)
        if derived > MAX_DERIVED_LOGPROBS:
            raise ReferenceProtocolError("reference derived logprob work exceeds its hard bound")
        object.__setattr__(self, "prompts", prompts)
        _request_payload_size(self)


@dataclass(frozen=True)
class ReferenceTokenEvidence:
    target_logprob: float
    true_argmax_token_id: int
    support_logprobs: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_logprob", _f32(self.target_logprob, "target logprob"))
        object.__setattr__(self, "true_argmax_token_id", _integer(self.true_argmax_token_id, "true argmax token", 0, MAX_TOKEN_ID))
        support = tuple(_f32(value, "support logprob") for value in _tuple(self.support_logprobs, "support logprobs"))
        if not support:
            raise ReferenceProtocolError("support logprobs must be nonempty")
        object.__setattr__(self, "support_logprobs", support)


@dataclass(frozen=True)
class ReferenceRoleEvidence:
    tokens: tuple[ReferenceTokenEvidence, ...]

    def __post_init__(self) -> None:
        rows = _tuple(self.tokens, "reference token evidence")
        if not rows or any(type(row) is not ReferenceTokenEvidence for row in rows):
            raise ReferenceProtocolError("reference token evidence is invalid")
        object.__setattr__(self, "tokens", rows)


@dataclass(frozen=True)
class ReferencePromptEvidence:
    prompt_digest: str
    prompt_token_count: int
    prompt_token_sha256: str
    roles: tuple[ReferenceRoleEvidence, ReferenceRoleEvidence, ReferenceRoleEvidence]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_digest", _digest(self.prompt_digest, "prompt digest"))
        object.__setattr__(self, "prompt_token_count", _integer(self.prompt_token_count, "prompt token count", 1, MAX_INDEX))
        object.__setattr__(self, "prompt_token_sha256", _digest(self.prompt_token_sha256, "prompt-token digest"))
        object.__setattr__(self, "roles", _roles(self.roles, ReferenceRoleEvidence, "evidence roles"))


@dataclass(frozen=True)
class ReferenceEvidence:
    session_id: str
    launch_digest: str
    plan_digest: str
    request_sha256: str
    request_id: str
    nonce: str
    request_index: int
    vocab_size: int
    prompts: tuple[ReferencePromptEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _binding(self.session_id, "session ID"))
        object.__setattr__(self, "launch_digest", _digest(self.launch_digest, "launch digest"))
        object.__setattr__(self, "plan_digest", _digest(self.plan_digest, "plan digest"))
        object.__setattr__(self, "request_sha256", _digest(self.request_sha256, "request digest"))
        object.__setattr__(self, "request_id", _binding(self.request_id, "request ID"))
        object.__setattr__(self, "nonce", _binding(self.nonce, "nonce"))
        object.__setattr__(self, "request_index", _integer(self.request_index, "request index", 0, MAX_INDEX))
        object.__setattr__(self, "vocab_size", _integer(self.vocab_size, "vocabulary size", 1, MAX_TOKEN_ID + 1))
        prompts = _tuple(self.prompts, "reference evidence prompts")
        if not prompts or any(type(row) is not ReferencePromptEvidence for row in prompts):
            raise ReferenceProtocolError("reference evidence prompt coverage is invalid")
        object.__setattr__(self, "prompts", prompts)


def _request_payload_size(request: ReferenceRequest) -> int:
    total = _REQUEST_HEADER.size
    geometry = _ROLE_COUNT * request.tokens_per_prompt * (1 + request.support_width) * _U32.size
    for prompt in request.prompts:
        total += _PROMPT_HEADER.size + len(prompt.prompt.encode("utf-8")) + geometry
    if total > MAX_REQUEST_BYTES:
        raise ReferenceProtocolError("reference request exceeds its hard byte bound")
    return total


def _frame(magic: bytes, payload: bytes, maximum: int) -> bytes:
    if len(payload) > maximum:
        raise ReferenceProtocolError("reference frame exceeds its hard byte bound")
    return magic + struct.pack(">I", len(payload)) + payload


def _payload(frame: bytes, magic: bytes, maximum: int, label: str) -> bytes:
    if not isinstance(frame, bytes) or len(frame) < FRAME_HEADER_BYTES:
        raise ReferenceProtocolError(f"{label} frame is truncated")
    if frame[:4] != magic:
        raise ReferenceProtocolError(f"{label} frame magic/version mismatch")
    size = struct.unpack(">I", frame[4:8])[0]
    if size > maximum or len(frame) != FRAME_HEADER_BYTES + size:
        raise ReferenceProtocolError(f"{label} frame has an invalid size")
    return frame[8:]


def encode_reference_request(request: ReferenceRequest) -> bytes:
    if type(request) is not ReferenceRequest:
        raise ReferenceProtocolError("reference request is not typed")
    payload = bytearray(_REQUEST_HEADER.pack(
        bytes.fromhex(request.session_id), bytes.fromhex(request.launch_digest),
        bytes.fromhex(request.plan_digest), bytes.fromhex(request.request_id),
        bytes.fromhex(request.nonce), request.request_index, len(request.prompts),
        request.tokens_per_prompt, request.support_width,
    ))
    for prompt in request.prompts:
        text = prompt.prompt.encode("utf-8")
        payload.extend(_PROMPT_HEADER.pack(bytes.fromhex(prompt.prompt_digest), len(text)))
        payload.extend(text)
        for role in prompt.roles:
            for token in role.output_ids:
                payload.extend(_U32.pack(token))
            for support in role.supports:
                for token in support:
                    payload.extend(_U32.pack(token))
    if len(payload) != _request_payload_size(request):  # pragma: no cover
        raise AssertionError("reference request encoder violated its size table")
    return _frame(REQUEST_MAGIC, bytes(payload), MAX_REQUEST_BYTES)


def decode_reference_request(frame: bytes) -> ReferenceRequest:
    payload = _payload(frame, REQUEST_MAGIC, MAX_REQUEST_BYTES, "reference request")
    if len(payload) < _REQUEST_HEADER.size:
        raise ReferenceProtocolError("reference request binding is truncated")
    (session, launch, plan, request_id, nonce, request_index, prompt_count,
     token_count, support_width) = _REQUEST_HEADER.unpack_from(payload)
    if not 1 <= prompt_count <= MAX_PROMPTS:
        raise ReferenceProtocolError("reference request prompt count is invalid")
    _integer(request_index, "request index", 0, MAX_INDEX)
    _integer(token_count, "token count", 1, MAX_TOKENS)
    _integer(support_width, "support width", 1, MAX_SUPPORT_WIDTH)
    offset = _REQUEST_HEADER.size

    def take(size: int) -> bytes:
        nonlocal offset
        if size < 0 or offset + size > len(payload):
            raise ReferenceProtocolError("reference request is truncated")
        result = payload[offset:offset + size]
        offset += size
        return result

    prompts = []
    for _ in range(prompt_count):
        prompt_digest, text_size = _PROMPT_HEADER.unpack(take(_PROMPT_HEADER.size))
        if text_size > MAX_PROMPT_BYTES:
            raise ReferenceProtocolError("reference prompt exceeds its byte bound")
        try:
            text = take(text_size).decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ReferenceProtocolError(f"reference prompt is not UTF-8: {exc}") from None
        roles = []
        for _role in range(_ROLE_COUNT):
            outputs = tuple(_U32.unpack(take(_U32.size))[0] for _ in range(token_count))
            supports = tuple(tuple(
                _U32.unpack(take(_U32.size))[0] for _ in range(support_width)
            ) for _ in range(token_count))
            roles.append(ReferenceRoleInput(outputs, supports))
        prompts.append(ReferencePromptInput(prompt_digest.hex(), text, tuple(roles)))
    if offset != len(payload):
        raise ReferenceProtocolError("reference request contains trailing bytes")
    result = ReferenceRequest(
        session.hex(), launch.hex(), plan.hex(), request_id.hex(), nonce.hex(),
        request_index, token_count, support_width, tuple(prompts),
    )
    if encode_reference_request(result) != frame:
        raise ReferenceProtocolError("reference request encoding is not canonical")
    return result


def request_sha256(request: ReferenceRequest) -> str:
    return sha256_hex(encode_reference_request(request))


def expected_evidence_payload_bytes(request: ReferenceRequest) -> int:
    if type(request) is not ReferenceRequest:
        raise ReferenceProtocolError("reference request is not typed")
    per_token = _TOKEN_EVIDENCE.size + request.support_width * _F32.size
    total = _EVIDENCE_HEADER.size + len(request.prompts) * (
        _EVIDENCE_PROMPT.size + _ROLE_COUNT * request.tokens_per_prompt * per_token
    )
    if total > MAX_EVIDENCE_BYTES:
        raise ReferenceProtocolError("reference evidence exceeds its hard byte bound")
    return total


def _bind_evidence(evidence: ReferenceEvidence, request: ReferenceRequest) -> None:
    expected = (
        request.session_id, request.launch_digest, request.plan_digest,
        request_sha256(request), request.request_id, request.nonce,
        request.request_index,
    )
    actual = (
        evidence.session_id, evidence.launch_digest, evidence.plan_digest,
        evidence.request_sha256, evidence.request_id, evidence.nonce,
        evidence.request_index,
    )
    if actual != expected or len(evidence.prompts) != len(request.prompts):
        raise ReferenceProtocolError("reference evidence binds another request")
    if tuple(row.prompt_digest for row in evidence.prompts) != tuple(
        row.prompt_digest for row in request.prompts
    ):
        raise ReferenceProtocolError("reference evidence prompt order differs from request")
    for prompt in evidence.prompts:
        for role in prompt.roles:
            if len(role.tokens) != request.tokens_per_prompt or any(
                len(token.support_logprobs) != request.support_width for token in role.tokens
            ):
                raise ReferenceProtocolError("reference evidence geometry differs from request")
            if any(token.true_argmax_token_id >= evidence.vocab_size for token in role.tokens):
                raise ReferenceProtocolError("reference evidence argmax exceeds vocabulary")


def encode_reference_evidence(evidence: ReferenceEvidence, request: ReferenceRequest) -> bytes:
    if type(evidence) is not ReferenceEvidence or type(request) is not ReferenceRequest:
        raise ReferenceProtocolError("reference evidence/request is not typed")
    _bind_evidence(evidence, request)
    payload = bytearray(_EVIDENCE_HEADER.pack(
        bytes.fromhex(evidence.session_id), bytes.fromhex(evidence.launch_digest),
        bytes.fromhex(evidence.plan_digest), bytes.fromhex(evidence.request_sha256),
        bytes.fromhex(evidence.request_id), bytes.fromhex(evidence.nonce),
        evidence.request_index, len(evidence.prompts), request.tokens_per_prompt,
        evidence.vocab_size, request.support_width,
    ))
    for prompt in evidence.prompts:
        payload.extend(_EVIDENCE_PROMPT.pack(
            bytes.fromhex(prompt.prompt_digest), prompt.prompt_token_count,
            bytes.fromhex(prompt.prompt_token_sha256),
        ))
        for role in prompt.roles:
            for token in role.tokens:
                payload.extend(_TOKEN_EVIDENCE.pack(
                    token.target_logprob, token.true_argmax_token_id
                ))
                for value in token.support_logprobs:
                    payload.extend(_F32.pack(value))
    if len(payload) != expected_evidence_payload_bytes(request):  # pragma: no cover
        raise AssertionError("reference evidence encoder violated its size table")
    return _frame(EVIDENCE_MAGIC, bytes(payload), MAX_EVIDENCE_BYTES)


def decode_reference_evidence(frame: bytes, request: ReferenceRequest) -> ReferenceEvidence:
    payload = _payload(frame, EVIDENCE_MAGIC, MAX_EVIDENCE_BYTES, "reference evidence")
    expected = expected_evidence_payload_bytes(request)
    if len(payload) != expected:
        raise ReferenceProtocolError("reference evidence has the wrong exact size")
    (session, launch, plan, request_digest, request_id, nonce, request_index,
     prompt_count, token_count, vocab_size, support_width) = _EVIDENCE_HEADER.unpack_from(payload)
    if (prompt_count, token_count, support_width) != (
        len(request.prompts), request.tokens_per_prompt, request.support_width
    ):
        raise ReferenceProtocolError("reference evidence geometry binding mismatch")
    offset = _EVIDENCE_HEADER.size
    prompts = []
    for _ in range(prompt_count):
        prompt_digest, prompt_tokens, prompt_token_digest = _EVIDENCE_PROMPT.unpack_from(payload, offset)
        offset += _EVIDENCE_PROMPT.size
        roles = []
        for _role in range(_ROLE_COUNT):
            tokens = []
            for _position in range(token_count):
                target, argmax = _TOKEN_EVIDENCE.unpack_from(payload, offset)
                offset += _TOKEN_EVIDENCE.size
                support = []
                for _support in range(support_width):
                    (value,) = _F32.unpack_from(payload, offset)
                    offset += _F32.size
                    support.append(value)
                tokens.append(ReferenceTokenEvidence(target, argmax, tuple(support)))
            roles.append(ReferenceRoleEvidence(tuple(tokens)))
        prompts.append(ReferencePromptEvidence(
            prompt_digest.hex(), prompt_tokens, prompt_token_digest.hex(), tuple(roles)
        ))
    if offset != len(payload):  # pragma: no cover - exact size already proves it
        raise ReferenceProtocolError("reference evidence contains trailing bytes")
    result = ReferenceEvidence(
        session.hex(), launch.hex(), plan.hex(), request_digest.hex(),
        request_id.hex(), nonce.hex(), request_index, vocab_size, tuple(prompts),
    )
    _bind_evidence(result, request)
    if encode_reference_evidence(result, request) != frame:
        raise ReferenceProtocolError("reference evidence encoding is not canonical")
    return result


__all__ = [
    "EVIDENCE_MAGIC", "FRAME_HEADER_BYTES", "MAX_EVIDENCE_BYTES",
    "MAX_DERIVED_LOGPROBS", "MAX_REQUEST_BYTES", "MAX_SUPPORT_UNION",
    "REQUEST_MAGIC", "ROLE_NAMES", "ReferenceEvidence",
    "ReferencePromptEvidence",
    "ReferencePromptInput", "ReferenceProtocolError", "ReferenceRequest",
    "ReferenceRoleEvidence", "ReferenceRoleInput", "ReferenceTokenEvidence",
    "decode_reference_evidence", "decode_reference_request",
    "encode_reference_evidence", "encode_reference_request",
    "expected_evidence_payload_bytes", "request_sha256",
]
